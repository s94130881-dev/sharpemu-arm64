// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Core.Memory;

public static class GuestMemoryConfiguration
{
    // 26 GiB
    public const ulong RamSize =
        26UL * 1024UL * 1024UL * 1024UL;

    // 26624 MiB
    public const ulong RamSizeMiB =
        26UL * 1024UL;

    // Número de páginas de 4 KiB
    public const ulong PageSize = 0x1000UL;

    public const ulong RamPages =
        RamSize / PageSize;

    public static ulong GetRamSize()
    {
        return RamSize;
    }

    public static ulong GetRamSizeMiB()
    {
        return RamSizeMiB;
    }

    public static ulong GetRamPages()
    {
        return RamPages;
    }
}
