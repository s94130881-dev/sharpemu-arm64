// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Core.Loader;
using SharpEmu.HLE;

namespace SharpEmu.Core.Memory;

public sealed class VirtualMemory : IVirtualMemory
{
    // Capacidade máxima de RAM virtual do convidado:
    // 26 GiB = 26 * 1024^3 bytes.
    public const ulong MaxGuestMemory = 26UL * 1024UL * 1024UL * 1024UL;

    // Páginas de 64 KiB.
    private const int PageSize = 64 * 1024;

    private readonly object _gate = new();
    private readonly List<MappedRegion> _regions = new();

    public void Clear()
    {
        lock (_gate)
        {
            _regions.Clear();
        }
    }

    public void Map(
        ulong virtualAddress,
        ulong memorySize,
        ulong fileOffset,
        ReadOnlySpan<byte> fileData,
        ProgramHeaderFlags protection)
    {
        if (memorySize == 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(memorySize),
                "Memory size must be greater than zero.");
        }

        if ((ulong)fileData.Length > memorySize)
        {
            throw new ArgumentOutOfRangeException(
                nameof(fileData),
                "File size cannot exceed memory size.");
        }

        var endAddress = checked(virtualAddress + memorySize);

        // Impede que o espaço virtual ultrapasse o limite configurado.
        if (endAddress > MaxGuestMemory)
        {
            throw new NotSupportedException(
                $"Virtual memory cannot exceed {MaxGuestMemory / (1024UL * 1024UL * 1024UL)} GiB.");
        }

        lock (_gate)
        {
            var insertionIndex = FindInsertionIndex(virtualAddress);

            if ((insertionIndex > 0 &&
                 virtualAddress < _regions[insertionIndex - 1].EndAddress) ||
                (insertionIndex < _regions.Count &&
                 endAddress > _regions[insertionIndex].Region.VirtualAddress))
            {
                throw new InvalidOperationException(
                    "Attempted to map an overlapping virtual memory region.");
            }

            var backingMemory = new SparseMemory(memorySize);

            // Copia somente os dados existentes no arquivo.
            if (!fileData.IsEmpty)
            {
                backingMemory.Write(0, fileData);
            }

            _regions.Insert(
                insertionIndex,
                new MappedRegion(
                    new VirtualMemoryRegion(
                        virtualAddress,
                        memorySize,
                        fileOffset,
                        (ulong)fileData.Length,
                        protection),
                    endAddress,
                    backingMemory));
        }
    }

    public IReadOnlyList<VirtualMemoryRegion> SnapshotRegions()
    {
        lock (_gate)
        {
            var snapshot = new VirtualMemoryRegion[_regions.Count];

            for (var i = 0; i < _regions.Count; i++)
            {
                snapshot[i] = _regions[i].Region;
            }

            return snapshot;
        }
    }

    public bool TryRead(
        ulong virtualAddress,
        Span<byte> destination)
    {
        lock (_gate)
        {
            if (!TryValidateRange(
                    virtualAddress,
                    destination.Length,
                    ProgramHeaderFlags.Read,
                    out var regionIndex))
            {
                return false;
            }

            CopyFromRegions(
                virtualAddress,
                destination,
                regionIndex);

            return true;
        }
    }

    public bool TryWrite(
        ulong virtualAddress,
        ReadOnlySpan<byte> source)
    {
        lock (_gate)
        {
            if (!TryValidateRange(
                    virtualAddress,
                    source.Length,
                    ProgramHeaderFlags.Write,
                    out var regionIndex))
            {
                return false;
            }

            CopyToRegions(
                virtualAddress,
                source,
                regionIndex);
        }

        if (GuestWriteWatch.Armed)
        {
            GuestWriteWatch.Check(
                virtualAddress,
                source);
        }

        return true;
    }

    private bool TryValidateRange(
        ulong virtualAddress,
        int length,
        ProgramHeaderFlags requiredProtection,
        out int regionIndex)
    {
        regionIndex =
            FindContainingRegionIndex(virtualAddress);

        if (regionIndex < 0)
        {
            return false;
        }

        var currentAddress = virtualAddress;
        var remaining = length;
        var currentIndex = regionIndex;

        while (true)
        {
            if (currentIndex >= _regions.Count)
            {
                return false;
            }

            var region = _regions[currentIndex];

            if (currentAddress < region.Region.VirtualAddress ||
                currentAddress >= region.EndAddress ||
                (region.Region.Protection & requiredProtection) == 0)
            {
                return false;
            }

            if (remaining == 0)
            {
                return true;
            }

            var available =
                region.EndAddress - currentAddress;

            var chunkLength =
                (int)Math.Min(
                    (ulong)remaining,
                    available);

            remaining -= chunkLength;

            if (remaining == 0)
            {
                return true;
            }

            currentAddress += (ulong)chunkLength;
            currentIndex++;
        }
    }

    private int FindContainingRegionIndex(
        ulong virtualAddress)
    {
        var insertionIndex =
            FindInsertionIndex(virtualAddress);

        if (insertionIndex < _regions.Count &&
            _regions[insertionIndex].Region.VirtualAddress ==
            virtualAddress)
        {
            return insertionIndex;
        }

        var candidateIndex =
            insertionIndex - 1;

        return candidateIndex >= 0 &&
               virtualAddress <
               _regions[candidateIndex].EndAddress
            ? candidateIndex
            : -1;
    }

    private void CopyFromRegions(
        ulong virtualAddress,
        Span<byte> destination,
        int regionIndex)
    {
        var copied = 0;
        var currentAddress = virtualAddress;

        while (copied < destination.Length)
        {
            var region = _regions[regionIndex++];

            var regionOffset =
                currentAddress -
                region.Region.VirtualAddress;

            var chunkLength =
                (int)Math.Min(
                    (ulong)(destination.Length - copied),
                    region.Region.MemorySize - regionOffset);

            region.BackingMemory.Read(
                regionOffset,
                destination.Slice(
                    copied,
                    chunkLength));

            copied += chunkLength;
            currentAddress += (ulong)chunkLength;
        }
    }

    private void CopyToRegions(
        ulong virtualAddress,
        ReadOnlySpan<byte> source,
        int regionIndex)
    {
        var copied = 0;
        var currentAddress = virtualAddress;

        while (copied < source.Length)
        {
            var region = _regions[regionIndex++];

            var regionOffset =
                currentAddress -
                region.Region.VirtualAddress;

            var chunkLength =
                (int)Math.Min(
                    (ulong)(source.Length - copied),
                    region.Region.MemorySize - regionOffset);

            region.BackingMemory.Write(
                regionOffset,
                source.Slice(
                    copied,
                    chunkLength));

            copied += chunkLength;
            currentAddress += (ulong)chunkLength;
        }
    }

    private int FindInsertionIndex(
        ulong virtualAddress)
    {
        var lower = 0;
        var upper = _regions.Count;

        while (lower < upper)
        {
            var middle =
                lower + ((upper - lower) / 2);

            if (_regions[middle]
                    .Region
                    .VirtualAddress < virtualAddress)
            {
                lower = middle + 1;
            }
            else
            {
                upper = middle;
            }
        }

        return lower;
    }

    private readonly record struct MappedRegion(
        VirtualMemoryRegion Region,
        ulong EndAddress,
        SparseMemory BackingMemory);

    /// <summary>
    /// Memória esparsa.
    ///
    /// O convidado pode possuir um espaço de até 26 GiB,
    /// mas somente as páginas efetivamente acessadas
    /// são alocadas no host.
    /// </summary>
    private sealed class SparseMemory
    {
        private readonly ulong _size;

        private readonly Dictionary<ulong, byte[]> _pages =
            new();

        public SparseMemory(ulong size)
        {
            _size = size;
        }

        public void Read(
            ulong offset,
            Span<byte> destination)
        {
            ValidateRange(
                offset,
                (ulong)destination.Length);

            var processed = 0;

            while (processed < destination.Length)
            {
                var currentOffset =
                    offset + (ulong)processed;

                var pageIndex =
                    currentOffset / PageSize;

                var pageOffset =
                    (int)(currentOffset % PageSize);

                var available =
                    PageSize - pageOffset;

                var count =
                    Math.Min(
                        available,
                        destination.Length - processed);

                if (_pages.TryGetValue(
                        pageIndex,
                        out var page))
                {
                    page.AsSpan(
                            pageOffset,
                            count)
                        .CopyTo(
                            destination.Slice(
                                processed,
                                count));
                }
                else
                {
                    // Página nunca escrita = zero.
                    destination.Slice(
                            processed,
                            count)
                        .Clear();
                }

                processed += count;
            }
        }

        public void Write(
            ulong offset,
            ReadOnlySpan<byte> source)
        {
            ValidateRange(
                offset,
                (ulong)source.Length);

            var processed = 0;

            while (processed < source.Length)
            {
                var currentOffset =
                    offset + (ulong)processed;

                var pageIndex =
                    currentOffset / PageSize;

                var pageOffset =
                    (int)(currentOffset % PageSize);

                var available =
                    PageSize - pageOffset;

                var count =
                    Math.Min(
                        available,
                        source.Length - processed);

                if (!_pages.TryGetValue(
                        pageIndex,
                        out var page))
                {
                    page = new byte[PageSize];
                    _pages.Add(pageIndex, page);
                }

                source.Slice(
                        processed,
                        count)
                    .CopyTo(
                        page.AsSpan(
                            pageOffset,
                            count));

                processed += count;
            }
        }

        private void ValidateRange(
            ulong offset,
            ulong length)
        {
            if (offset > _size ||
                length > _size - offset)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(offset));
            }
        }
    }
}
