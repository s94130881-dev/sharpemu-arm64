// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Core.Loader;
using SharpEmu.HLE;

namespace SharpEmu.Core.Memory;

/// <summary>
/// Memória virtual do guest.
///
/// IMPORTANTE:
/// - GuestRamSize = 26 GiB é a capacidade configurada do guest.
/// - Não significa que 26 GiB de RAM física serão consumidos pelo host.
/// - A memória de cada região é armazenada de forma esparsa.
/// - Somente páginas realmente escritas são alocadas.
/// </summary>
public sealed class VirtualMemory : IVirtualMemory
{
    /// <summary>
    /// Quantidade de RAM apresentada ao guest.
    ///
    /// 26 GiB = 26 * 1024^3 bytes.
    /// </summary>
    public const ulong GuestRamSize =
        26UL * 1024UL * 1024UL * 1024UL;

    /// <summary>
    /// RAM em MiB.
    /// </summary>
    public const ulong GuestRamSizeMiB =
        26UL * 1024UL;

    /// <summary>
    /// RAM em GiB.
    /// </summary>
    public const ulong GuestRamSizeGiB = 26UL;

    /// <summary>
    /// Tamanho das páginas utilizadas pelo backing store.
    ///
    /// 64 KiB reduz o número de entradas do dicionário
    /// quando grandes quantidades de memória são utilizadas.
    /// </summary>
    private const int PageSize = 64 * 1024;

    private readonly object _gate = new();

    private readonly List<MappedRegion> _regions = new();

    /// <summary>
    /// Remove todos os mapeamentos.
    /// </summary>
    public void Clear()
    {
        lock (_gate)
        {
            _regions.Clear();
        }
    }

    /// <summary>
    /// Mapeia uma região de memória virtual do guest.
    ///
    /// O endereço virtual NÃO é limitado a 26 GiB.
    ///
    /// Os 26 GiB representam a capacidade de RAM do guest,
    /// enquanto o espaço de endereçamento virtual pode possuir
    /// endereços muito maiores.
    /// </summary>
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

        ulong endAddress;

        try
        {
            endAddress = checked(
                virtualAddress + memorySize);
        }
        catch (OverflowException)
        {
            throw new ArgumentOutOfRangeException(
                nameof(memorySize),
                "Virtual memory address range overflowed.");
        }

        lock (_gate)
        {
            var insertionIndex =
                FindInsertionIndex(virtualAddress);

            /*
             * Verifica sobreposição com a região anterior.
             */
            if (insertionIndex > 0 &&
                virtualAddress <
                _regions[insertionIndex - 1].EndAddress)
            {
                throw new InvalidOperationException(
                    "Attempted to map an overlapping virtual memory region.");
            }

            /*
             * Verifica sobreposição com a próxima região.
             */
            if (insertionIndex < _regions.Count &&
                endAddress >
                _regions[insertionIndex]
                    .Region
                    .VirtualAddress)
            {
                throw new InvalidOperationException(
                    "Attempted to map an overlapping virtual memory region.");
            }

            /*
             * Não criamos:
             *
             *     new byte[(int)memorySize]
             *
             * porque uma região pode ser muito maior que 2 GiB.
             *
             * Em vez disso utilizamos memória esparsa.
             */
            var backingMemory =
                new SparseMemory(memorySize);

            /*
             * Carrega os dados iniciais do ELF/arquivo.
             */
            if (!fileData.IsEmpty)
            {
                backingMemory.Write(
                    0,
                    fileData);
            }

            var mappedRegion =
                new MappedRegion(
                    new VirtualMemoryRegion(
                        virtualAddress,
                        memorySize,
                        fileOffset,
                        (ulong)fileData.Length,
                        protection),
                    endAddress,
                    backingMemory);

            _regions.Insert(
                insertionIndex,
                mappedRegion);
        }
    }

    /// <summary>
    /// Retorna uma cópia dos mapeamentos atuais.
    /// </summary>
    public IReadOnlyList<VirtualMemoryRegion>
        SnapshotRegions()
    {
        lock (_gate)
        {
            var snapshot =
                new VirtualMemoryRegion[_regions.Count];

            for (var i = 0;
                 i < _regions.Count;
                 i++)
            {
                snapshot[i] =
                    _regions[i].Region;
            }

            return snapshot;
        }
    }

    /// <summary>
    /// Lê memória do guest.
    /// </summary>
    public bool TryRead(
        ulong virtualAddress,
        Span<byte> destination)
    {
        if (destination.Length == 0)
        {
            return true;
        }

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

    /// <summary>
    /// Escreve memória do guest.
    /// </summary>
    public bool TryWrite(
        ulong virtualAddress,
        ReadOnlySpan<byte> source)
    {
        if (source.Length == 0)
        {
            return true;
        }

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

        /*
         * O write watch deve acontecer fora do lock
         * para evitar manter o lock durante callbacks.
         */
        if (GuestWriteWatch.Armed)
        {
            GuestWriteWatch.Check(
                virtualAddress,
                source);
        }

        return true;
    }

    /// <summary>
    /// Valida se uma faixa inteira está mapeada e possui
    /// a proteção necessária.
    /// </summary>
    private bool TryValidateRange(
        ulong virtualAddress,
        int length,
        ProgramHeaderFlags requiredProtection,
        out int regionIndex)
    {
        regionIndex =
            FindContainingRegionIndex(
                virtualAddress);

        if (regionIndex < 0)
        {
            return false;
        }

        if (length == 0)
        {
            return true;
        }

        ulong currentAddress =
            virtualAddress;

        ulong remaining =
            (ulong)length;

        var currentIndex =
            regionIndex;

        while (remaining > 0)
        {
            if (currentIndex >= _regions.Count)
            {
                return false;
            }

            var region =
                _regions[currentIndex];

            /*
             * O endereço precisa estar dentro da região.
             */
            if (currentAddress <
                    region.Region.VirtualAddress ||
                currentAddress >=
                    region.EndAddress)
            {
                return false;
            }

            /*
             * Verifica proteção.
             */
            if ((region.Region.Protection &
                 requiredProtection) == 0)
            {
                return false;
            }

            var available =
                region.EndAddress -
                currentAddress;

            var chunkLength =
                Math.Min(
                    remaining,
                    available);

            remaining -= chunkLength;

            if (remaining == 0)
            {
                return true;
            }

            currentAddress +=
                chunkLength;

            currentIndex++;
        }

        return true;
    }

    /// <summary>
    /// Localiza a região que contém um endereço virtual.
    /// </summary>
    private int FindContainingRegionIndex(
        ulong virtualAddress)
    {
        var insertionIndex =
            FindInsertionIndex(
                virtualAddress);

        /*
         * Endereço coincide exatamente com
         * o início de uma região.
         */
        if (insertionIndex < _regions.Count &&
            _regions[insertionIndex]
                .Region
                .VirtualAddress ==
            virtualAddress)
        {
            return insertionIndex;
        }

        /*
         * Caso contrário, a região candidata
         * é a anterior.
         */
        var candidateIndex =
            insertionIndex - 1;

        if (candidateIndex < 0)
        {
            return -1;
        }

        return virtualAddress <
               _regions[candidateIndex]
                   .EndAddress
            ? candidateIndex
            : -1;
    }

    /// <summary>
    /// Copia dados das regiões para o buffer do host.
    /// </summary>
    private void CopyFromRegions(
        ulong virtualAddress,
        Span<byte> destination,
        int regionIndex)
    {
        var copied = 0;

        ulong currentAddress =
            virtualAddress;

        while (copied < destination.Length)
        {
            if (regionIndex >= _regions.Count)
            {
                throw new InvalidOperationException(
                    "Memory range became invalid while reading.");
            }

            var region =
                _regions[regionIndex];

            var regionOffset =
                checked(
                    currentAddress -
                    region.Region.VirtualAddress);

            var available =
                region.Region.MemorySize -
                regionOffset;

            var requested =
                (ulong)(
                    destination.Length -
                    copied);

            var chunkLength =
                (int)Math.Min(
                    available,
                    requested);

            if (chunkLength <= 0)
            {
                throw new InvalidOperationException(
                    "Invalid memory region while reading.");
            }

            region.BackingMemory.Read(
                regionOffset,
                destination.Slice(
                    copied,
                    chunkLength));

            copied += chunkLength;

            currentAddress +=
                (ulong)chunkLength;

            /*
             * Se ainda há dados, passamos
             * para a próxima região.
             */
            if (copied < destination.Length)
            {
                regionIndex++;
            }
        }
    }

    /// <summary>
    /// Copia dados do host para as regiões do guest.
    /// </summary>
    private void CopyToRegions(
        ulong virtualAddress,
        ReadOnlySpan<byte> source,
        int regionIndex)
    {
        var copied = 0;

        ulong currentAddress =
            virtualAddress;

        while (copied < source.Length)
        {
            if (regionIndex >= _regions.Count)
            {
                throw new InvalidOperationException(
                    "Memory range became invalid while writing.");
            }

            var region =
                _regions[regionIndex];

            var regionOffset =
                checked(
                    currentAddress -
                    region.Region.VirtualAddress);

            var available =
                region.Region.MemorySize -
                regionOffset;

            var requested =
                (ulong)(
                    source.Length -
                    copied);

            var chunkLength =
                (int)Math.Min(
                    available,
                    requested);

            if (chunkLength <= 0)
            {
                throw new InvalidOperationException(
                    "Invalid memory region while writing.");
            }

            region.BackingMemory.Write(
                regionOffset,
                source.Slice(
                    copied,
                    chunkLength));

            copied += chunkLength;

            currentAddress +=
                (ulong)chunkLength;

            if (copied < source.Length)
            {
                regionIndex++;
            }
        }
    }

    /// <summary>
    /// Pesquisa binária para localizar a posição
    /// de uma região.
    /// </summary>
    private int FindInsertionIndex(
        ulong virtualAddress)
    {
        var lower = 0;
        var upper = _regions.Count;

        while (lower < upper)
        {
            var middle =
                lower +
                ((upper - lower) / 2);

            if (_regions[middle]
                    .Region
                    .VirtualAddress <
                virtualAddress)
            {
                lower =
                    middle + 1;
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
    /// Backing store de memória esparsa.
    ///
    /// Uma região pode representar dezenas de GiB,
    /// mas páginas não utilizadas não são alocadas.
    /// </summary>
    private sealed class SparseMemory
    {
        private readonly ulong _size;

        /*
         * Cada entrada representa uma página realmente
         * utilizada pelo guest.
         */
        private readonly Dictionary<
            ulong,
            byte[]> _pages = new();

        public SparseMemory(ulong size)
        {
            if (size == 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(size));
            }

            _size = size;
        }

        /// <summary>
        /// Lê bytes da memória esparsa.
        ///
        /// Páginas que nunca foram escritas
        /// retornam zero.
        /// </summary>
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
                    offset +
                    (ulong)processed;

                var pageIndex =
                    currentOffset /
                    (ulong)PageSize;

                var pageOffset =
                    (int)(
                        currentOffset %
                        (ulong)PageSize);

                var available =
                    PageSize -
                    pageOffset;

                var count =
                    Math.Min(
                        available,
                        destination.Length -
                        processed);

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
                    /*
                     * Página nunca utilizada:
                     * comportamento equivalente a
                     * memória zerada.
                     */
                    destination.Slice(
                            processed,
                            count)
                        .Clear();
                }

                processed += count;
            }
        }

        /// <summary>
        /// Escreve bytes na memória esparsa.
        ///
        /// Uma nova página só é criada quando
        /// realmente recebe dados.
        /// </summary>
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
                    offset +
                    (ulong)processed;

                var pageIndex =
                    currentOffset /
                    (ulong)PageSize;

                var pageOffset =
                    (int)(
                        currentOffset %
                        (ulong)PageSize);

                var available =
                    PageSize -
                    pageOffset;

                var count =
                    Math.Min(
                        available,
                        source.Length -
                        processed);

                if (!_pages.TryGetValue(
                        pageIndex,
                        out var page))
                {
                    page = new byte[PageSize];

                    _pages.Add(
                        pageIndex,
                        page);
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

        /// <summary>
        /// Valida uma operação de memória.
        /// </summary>
        private void ValidateRange(
            ulong offset,
            ulong length)
        {
            /*
             * Esta forma evita overflow em:
             *
             * offset + length
             */
            if (offset > _size ||
                length > _size - offset)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(offset),
                    "Memory access is outside the mapped region.");
            }
        }
    }
}
