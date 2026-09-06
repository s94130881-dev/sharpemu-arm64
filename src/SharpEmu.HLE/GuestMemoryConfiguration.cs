// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Core.Memory;

public static class GuestMemoryConfiguration
{
    // 26 GiB = 27917287424 bytes
    public const ulong RamSize =
        26UL * 1024UL * 1024UL * 1024UL;

    public const ulong RamSizeMiB =
        RamSize / (1024UL * 1024UL);

    public const ulong RamSizeGiB = 26UL;

    public const ulong PageSize = 0x1000UL;

    public static ulong PageCount =>
        RamSize / PageSize;
}
