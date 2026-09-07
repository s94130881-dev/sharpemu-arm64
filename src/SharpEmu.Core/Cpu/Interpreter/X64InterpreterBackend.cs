// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using Iced.Intel;
using SharpEmu.Core.Cpu.Disasm;
using SharpEmu.HLE;
using SharpEmu.Libs.Kernel;

namespace SharpEmu.Core.Cpu.Interpreter;

public sealed class X64InterpreterBackend
{
    private const int MaxInstructionBytes = 15;
    private readonly IModuleManager _moduleManager;
    private readonly X64InterpreterGuestThreadScheduler? _blockRegistry;
    // Ring buffer instead of Queue<T>: PushRecent runs unconditionally on every single instruction
    // executed (the crash-diagnostics payoff — "what ran right before this fault" — is only ever read
    // on the rare failure path), so its steady-state cost is Queue<T>'s Enqueue/Dequeue bookkeeping
    // paid millions of times for a benefit realized maybe once. A fixed-size array + wrapping index is
    // the same FIFO-of-last-256-entries behavior (FormatRecentInstructions below preserves the exact
    // oldest-to-newest ordering Queue<T>'s enumerator produced) at a fraction of the per-instruction cost.
    private readonly (Instruction Instruction, byte[] Bytes)[] _recentInstructions = new (Instruction, byte[])[RecentInstructionCapacity];
    private int _recentInstructionHead;
    private int _recentInstructionCount;

    // Decoded-instruction cache keyed by guest address. Decoding x86-64 with
    // Iced is by far the most expensive part of the per-instruction loop, so
    // a hot loop that executes the same handful of addresses millions of
    // times previously re-decoded every single time. A cache hit is only
    // accepted when the freshly-read guest bytes still match what was
    // decoded (checked every time, via the now-batched memory read), so a
    // stale entry is never used after the guest rewrites/recompiles code at
    // that address â€” this backend is instance-per-guest-thread (see
    // X64InterpreterGuestThreadScheduler), so unlike a shared interpreter
    // instance this cache needs no locking or thread-local storage.
    // A real game's hot working set spans far more than one tight loop -- rendering, audio, and
    // game-logic subsystems each contribute their own hot addresses on the SAME per-thread cache,
    // so 13 bits (8192 entries) collides often enough to force expensive Iced re-decodes on
    // addresses that should stay cached. 16 bits (65536 entries, ~4-5MB per guest thread at this
    // struct's size) trades a modest, one-time memory cost for meaningfully fewer cache misses.
    private const int DecodeCacheBits = 16;
    private const int DecodeCacheSize = 1 << DecodeCacheBits;
    private readonly DecodeCacheEntry[] _decodeCache = new DecodeCacheEntry[DecodeCacheSize];

    private struct DecodeCacheEntry
    {
        public bool Valid;
        public ulong Tag;
        public byte[]? Bytes;
        public Instruction Instruction;

        // Set once, at cache-fill time, when the fetched bytes came from a region that cannot be
        // written by ordinary guest stores (self-modifying code is architecturally impossible
        // there). A hit against such an entry can skip re-reading/re-validating the guest bytes
        // entirely for as long as MappingGeneration stays at CachedMappingGeneration -- see
        // ICpuMemory.MappingGeneration's doc comment for why that pairing is safe.
        public bool IsNonWritable;
        public long CachedMappingGeneration;
    }

    public X64InterpreterBackend(IModuleManager moduleManager, X64InterpreterGuestThreadScheduler? blockRegistry = null)
    {
        _moduleManager = moduleManager ?? throw new ArgumentNullException(nameof(moduleManager));
        _blockRegistry = blockRegistry;
    }

    public X64InterpreterResult Execute(
        CpuContext context,
        ulong entryPoint,
        IReadOnlyDictionary<ulong, string> importStubs,
        X64InterpreterOptions options)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(importStubs);

        // CpuDispatcher's caller always builds effectiveImportStubs as a concrete Dictionary<ulong,
        // string> before this call, but the parameter above stays IReadOnlyDictionary for interface
        // stability (INativeCpuBackend/DirectExecutionBackend share the same shape). Every single
        // executed instruction probes this map once (the "did RIP just land on an import stub" check
        // below), so resolving the concrete type once here — rather than going through the interface
        // vtable millions of times — lets the JIT call/inline Dictionary<TKey,TValue>.TryGetValue
        // directly. This matters more on Android's Mono JIT, which (unlike desktop RyuJIT's PGO-driven
        // speculative devirtualization) won't do this on its own.
        var concreteImportStubs = importStubs as Dictionary<ulong, string>;

        context.Rip = entryPoint;
        var importsHit = 0;
        var uniqueImports = new HashSet<string>(StringComparer.Ordinal);
        var trace = options.Trace ? new StringBuilder(4096) : null;
        var instructionLimit = options.MaxInstructions < 0 ? 0 : options.MaxInstructions;

        for (var executed = 0; instructionLimit == 0 || executed < instructionLimit; executed++)
        {
            // Diagnostic heartbeat: a long-running guest thread that never
            // logs anything else looks identical (from the outside) whether
            // it's doing real, slow, interpreted work or spinning on an
            // unimplemented synchronization primitive. Print roughly twice
            // a second (instruction-count-based, so it's independent of
            // host clock) so RIP movement (real work) can be told apart
            // from RIP oscillating across a tiny address set (a spin loop).
            if (executed != 0 && (executed & 0xFFFFF) == 0)
            {
                Console.Error.WriteLine(
                    $"[CPU-INTERP][HEARTBEAT] thread={Thread.CurrentThread.Name ?? "primary"} executed={executed} rip=0x{context.Rip:X16}");
            }

            if (context.Rip == 0)
            {
                return Complete(
                    OrbisGen2Result.ORBIS_GEN2_OK,
                    CpuExitReason.ReturnedToHost,
                    context.Rip,
                    executed,
                    importsHit,
                    uniqueImports.Count,
                    trace);
            }

            string? nid;
            bool isImportStub;
            if (concreteImportStubs is not null)
            {
                isImportStub = concreteImportStubs.TryGetValue(context.Rip, out nid);
            }
            else
            {
                isImportStub = importStubs.TryGetValue(context.Rip, out nid);
            }

            if (isImportStub)
            {
                // TryGetValue returning true guarantees nid is non-null; the compiler can't see that
                // across the manual if/else branches above (it can for the direct out-var pattern this
                // replaced), hence the forgiveness operator here rather than a real nullability gap.
                importsHit++;
                uniqueImports.Add(nid!);
                trace?.AppendLine($"[CPU-INTERP][IMPORT] rip=0x{context.Rip:X16} nid={nid}");

                _ = _moduleManager.Dispatch(nid!, context);

                if (GuestThreadExecution.TryConsumeCurrentEntryExit(out var exitValue, out var exitReason))
                {
                    trace?.AppendLine($"[CPU-INTERP][EXIT] rip=0x{context.Rip:X16} reason={exitReason} value=0x{exitValue:X16}");
                    context[CpuRegister.Rax] = exitValue;
                    return Complete(
                        OrbisGen2Result.ORBIS_GEN2_OK,
                        CpuExitReason.Exited,
                        context.Rip,
                        executed,
                        importsHit,
                        uniqueImports.Count,
                        trace);
                }

                // The HLE export just requested a cooperative block (e.g.
                // sceKernelWaitSema with nothing to acquire yet). Actually
                // block this real host thread on the wake key rather than
                // letting the request go unhonored â€” unhonored, the guest
                // code sees an immediate "success" return and spins its own
                // wait loop as fast as the interpreter can execute it.
                if (GuestThreadExecution.TryConsumeCurrentThreadBlock(
                        out _,
                        out _,
                        out _,
                        out var wakeKey,
                        out var waiter,
                        out var blockDeadline))
                {
                    _blockRegistry?.TryBlockCurrentThread(wakeKey, blockDeadline);
                    if (waiter is not null)
                    {
                        context[CpuRegister.Rax] = unchecked((ulong)waiter.Resume());
                    }
                }

                if (!context.PopUInt64(out var returnAddress))
                {
                    return MemoryFault(
                        context,
                        context.Rip,
                        opcode: null,
                        context[CpuRegister.Rsp],
                        sizeof(ulong),
                        isWrite: false,
                        executed,
                        importsHit,
                        uniqueImports.Count,
                        trace);
                }

                context.Rip = returnAddress;
                continue;
            }

            if (!TryDecodeCached(context, context.Rip, out var instruction, out var bytes))
            {
                return NotImplemented(
                    context.Rip,
                    "decode_failed",
                    FormatBytes(bytes),
                    executed,
                    importsHit,
                    uniqueImports.Count,
                    trace);
            }

            var oldRip = context.Rip;
            PushRecent(instruction, bytes);
            if (trace is not null)
            {
                trace.AppendLine(FormatInstruction(instruction, bytes));
            }

            if (!TryExecuteInstruction(context, instruction, out var changedRip, out var failure))
            {
                return failure.Kind switch
                {
                    InterpreterFailureKind.MemoryRead => MemoryFault(
                        context,
                        oldRip,
                        bytes.Length > 0 ? bytes[0] : null,
                        failure.Address,
                        failure.Size,
                        isWrite: false,
                        executed,
                        importsHit,
                        uniqueImports.Count,
                        trace),
                    InterpreterFailureKind.MemoryWrite => MemoryFault(
                        context,
                        oldRip,
                        bytes.Length > 0 ? bytes[0] : null,
                        failure.Address,
                        failure.Size,
                        isWrite: true,
                        executed,
                        importsHit,
                        uniqueImports.Count,
                        trace),
                    InterpreterFailureKind.Trap => Trap(
                        oldRip,
                        bytes.Length > 0 ? bytes[0] : (byte)0xCC,
                        executed,
                        importsHit,
                        uniqueImports.Count,
                        trace),
                    _ => NotImplemented(
                        oldRip,
                        instruction.Mnemonic.ToString(),
                        failure.Detail,
                        executed,
                        importsHit,
                        uniqueImports.Count,
                        trace),
                };
            }

            if (!changedRip)
            {
                context.Rip = oldRip + (uint)instruction.Length;
            }
        }

        return Complete(
            OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_IMPLEMENTED,
            CpuExitReason.BudgetExceeded,
            context.Rip,
            instructionLimit,
            importsHit,
            uniqueImports.Count,
            trace,
            notImplementedInfo: new CpuNotImplementedInfo(
                CpuNotImplementedSource.InstructionBudget,
                context.Rip,
                nid: null,
                exportName: "x64_interpreter_budget",
                libraryName: "SharpEmu.Core",
                detail: $"Interpreter instruction budget exceeded ({instructionLimit})."));
    }

    private bool TryDecodeCached(CpuContext context, ulong rip, out Instruction instruction, out byte[] bytes)
    {
        instruction = default;

        var slot = (int)((rip * 0x9E3779B97F4A7C15UL) >> (64 - DecodeCacheBits));
        ref var entry = ref _decodeCache[slot];

        // Fast path: this exact address was cached from a region ordinary guest stores cannot
        // write to, so self-modifying code is architecturally impossible there -- as long as
        // nothing anywhere has been mapped/unmapped/reprotected since (MappingGeneration
        // unchanged), the cached bytes/instruction cannot have gone stale, so this skips the
        // guest-memory read this method otherwise does on every single execution, hit or miss.
        // See ICpuMemory.MappingGeneration's doc comment for the full safety argument.
        if (entry.Valid && entry.Tag == rip && entry.IsNonWritable &&
            context.Memory.MappingGeneration == entry.CachedMappingGeneration)
        {
            instruction = entry.Instruction;
            bytes = entry.Bytes!;
            return true;
        }

        // The fresh-bytes read is on the hot path of every single instruction
        // executed (cache hit or miss), so it reads into a stack buffer rather
        // than a freshly heap-allocated array â€” a decode-cache *hit* (the
        // overwhelming majority of executions inside any loop) only needs
        // these bytes transiently, to confirm the cached entry still matches
        // (self-modifying-code guard), and never needs to keep them. This
        // turns what was one small array allocation per instruction executed
        // into zero allocations on every cache hit; a genuine cache miss still
        // allocates once, to persist the bytes into the new cache entry.
        Span<byte> freshBytes = stackalloc byte[MaxInstructionBytes];
        if (!IcedDecoder.TryReadGuestBytes(context.Memory, rip, MaxInstructionBytes, freshBytes, out var freshLength))
        {
            bytes = Array.Empty<byte>();
            return false;
        }

        var validBytes = freshBytes[..freshLength];
        if (entry.Valid && entry.Tag == rip && DecodedBytesStillMatch(entry, validBytes))
        {
            instruction = entry.Instruction;
            bytes = entry.Bytes!;
            return true;
        }

        bytes = validBytes.ToArray();
        if (!TryDecodeRaw(rip, bytes, out instruction))
        {
            return false;
        }

        entry.Valid = true;
        entry.Tag = rip;
        entry.Bytes = bytes;
        entry.Instruction = instruction;
        entry.IsNonWritable = context.Memory.TryIsRegionNonWritable(rip);
        entry.CachedMappingGeneration = context.Memory.MappingGeneration;
        return true;
    }

    private static bool DecodedBytesStillMatch(in DecodeCacheEntry entry, ReadOnlySpan<byte> freshBytes)
    {
        var length = entry.Instruction.Length;
        var cached = entry.Bytes;
        if (cached is null || cached.Length < length || freshBytes.Length < length)
        {
            return false;
        }

        return cached.AsSpan(0, length).SequenceEqual(freshBytes[..length]);
    }

    private static bool TryDecodeRaw(ulong rip, byte[] bytes, out Instruction instruction)
    {
        instruction = default;
        try
        {
            var decoder = Iced.Intel.Decoder.Create(64, new ByteArrayCodeReader(bytes));
            decoder.IP = rip;
            decoder.Decode(out instruction);
            return instruction.Code != Code.INVALID && instruction.Length > 0;
        }
        catch
        {
            return false;
        }
    }

    private bool TryExecuteInstruction(
        CpuContext context,
        in Instruction instruction,
        out bool changedRip,
        out InterpreterFailure failure)
    {
        changedRip = false;
        failure = default;

        switch (instruction.Mnemonic)
        {
            case Mnemonic.Nop:
            case Mnemonic.Pause:
                return true;
            case Mnemonic.Int3:
            case Mnemonic.Hlt:
                failure = InterpreterFailure.Trap();
                return false;
            case Mnemonic.Ret:
                if (!context.PopUInt64(out var retTarget))
                {
                    failure = InterpreterFailure.MemoryRead(context[CpuRegister.Rsp], sizeof(ulong));
                    return false;
                }
                context.Rip = retTarget;
                changedRip = true;
                return true;
            case Mnemonic.Call:
                if (!TryGetBranchTarget(context, instruction, out var callTarget, out failure))
                {
                    return false;
                }
                if (!context.PushUInt64(context.Rip + (uint)instruction.Length))
                {
                    failure = InterpreterFailure.MemoryWrite(context[CpuRegister.Rsp], sizeof(ulong));
                    return false;
                }
                context.Rip = callTarget;
                changedRip = true;
                return true;
            case Mnemonic.Jmp:
                if (!TryGetBranchTarget(context, instruction, out var jmpTarget, out failure))
                {
                    return false;
                }
                context.Rip = jmpTarget;
                changedRip = true;
                return true;
        }

        if (instruction.FlowControl == FlowControl.ConditionalBranch)
        {
            if (!X64Flags.TryEvaluateCondition(instruction.Mnemonic, context.Rflags, out var branchTaken))
            {
                failure = InterpreterFailure.Unsupported($"unsupported conditional branch {instruction.Mnemonic}");
                return false;
            }
            if (branchTaken)
            {
                context.Rip = instruction.NearBranch64;
                changedRip = true;
            }
            return true;
        }

        return instruction.Mnemonic switch
        {
            Mnemonic.Mov => ExecuteMove(context, instruction, out failure),
            Mnemonic.Movzx => ExecuteMovzx(context, instruction, out failure),
            Mnemonic.Movsx => ExecuteMovsx(context, instruction, out failure),
            Mnemonic.Movsxd => ExecuteMovsxd(context, instruction, out failure),
            Mnemonic.Cmovb => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovae => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovbe => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmova => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmove => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovne => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovs => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovns => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovo => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovno => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovl => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovge => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovle => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Cmovg => ExecuteConditionalMove(context, instruction, out failure),
            Mnemonic.Sete => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Setne => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Setb => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Setae => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Setbe => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Seta => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Sets => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Setns => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Seto => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Setno => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Setl => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Setge => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Setle => ExecuteSetCondition(context, instruction, out failure),
            Mnemonic.Setg => ExecuteSetCondition(context, instruction, out failure),
            // Legacy SSE/SSE2 forms. The guest is x86-64 even when the host is ARM64.
            // Keep these separate from the VEX forms because legacy two-operand
            // instructions use DEST as the first arithmetic source.
            Mnemonic.Movups => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Movupd => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Movaps => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Movapd => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Movdqu => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Movdqa => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Movss => ExecuteMoveScalarLegacy(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Movsd => ExecuteMoveScalarLegacy(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Movhps => ExecuteVectorMoveHighPackedSingleLegacy(context, instruction, out failure),
            Mnemonic.Movlps => ExecuteVectorMoveLowPackedSingleLegacy(context, instruction, out failure),
            Mnemonic.Movhlps => ExecuteMovHlps(context, instruction, out failure),
            Mnemonic.Movlhps => ExecuteMovLhps(context, instruction, out failure),
            Mnemonic.Sqrtss => ExecuteSqrt(context, instruction, doublePrecision: false, packed: false, out failure),
            Mnemonic.Sqrtsd => ExecuteSqrt(context, instruction, doublePrecision: true, packed: false, out failure),
            Mnemonic.Sqrtps => ExecuteSqrt(context, instruction, doublePrecision: false, packed: true, out failure),
            Mnemonic.Sqrtpd => ExecuteSqrt(context, instruction, doublePrecision: true, packed: true, out failure),
            Mnemonic.Rsqrtps => ExecuteReciprocalSqrt(context, instruction, out failure),
            Mnemonic.Rcpps => ExecuteReciprocal(context, instruction, out failure),
            Mnemonic.Rsqrtss => ExecuteReciprocalSqrtScalar(context, instruction, out failure),
            Mnemonic.Rcpss => ExecuteReciprocalScalar(context, instruction, out failure),
            Mnemonic.Movd => ExecuteVectorMoveD(context, instruction, out failure),
            Mnemonic.Movq => ExecuteVectorMoveQ(context, instruction, out failure),
            Mnemonic.Cvtsi2ss => ExecuteConvertIntToScalar(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Cvtsi2sd => ExecuteConvertIntToScalar(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Cvttss2si => ExecuteConvertScalarToInt(context, instruction, doublePrecision: false, truncate: true, out failure),
            Mnemonic.Cvttsd2si => ExecuteConvertScalarToInt(context, instruction, doublePrecision: true, truncate: true, out failure),
            Mnemonic.Cvtss2si => ExecuteConvertScalarToInt(context, instruction, doublePrecision: false, truncate: false, out failure),
            Mnemonic.Cvtsd2si => ExecuteConvertScalarToInt(context, instruction, doublePrecision: true, truncate: false, out failure),
            Mnemonic.Addss => ExecuteScalarArithLegacy(context, instruction, ScalarArithOp.Add, doublePrecision: false, out failure),
            Mnemonic.Subss => ExecuteScalarArithLegacy(context, instruction, ScalarArithOp.Sub, doublePrecision: false, out failure),
            Mnemonic.Mulss => ExecuteScalarArithLegacy(context, instruction, ScalarArithOp.Mul, doublePrecision: false, out failure),
            Mnemonic.Divss => ExecuteScalarArithLegacy(context, instruction, ScalarArithOp.Div, doublePrecision: false, out failure),
            Mnemonic.Addsd => ExecuteScalarArithLegacy(context, instruction, ScalarArithOp.Add, doublePrecision: true, out failure),
            Mnemonic.Subsd => ExecuteScalarArithLegacy(context, instruction, ScalarArithOp.Sub, doublePrecision: true, out failure),
            Mnemonic.Mulsd => ExecuteScalarArithLegacy(context, instruction, ScalarArithOp.Mul, doublePrecision: true, out failure),
            Mnemonic.Divsd => ExecuteScalarArithLegacy(context, instruction, ScalarArithOp.Div, doublePrecision: true, out failure),
            Mnemonic.Addps => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Add, doublePrecision: false, out failure),
            Mnemonic.Subps => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Sub, doublePrecision: false, out failure),
            Mnemonic.Mulps => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Mul, doublePrecision: false, out failure),
            Mnemonic.Divps => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Div, doublePrecision: false, out failure),
            Mnemonic.Minps => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Min, doublePrecision: false, out failure),
            Mnemonic.Maxps => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Max, doublePrecision: false, out failure),
            Mnemonic.Addpd => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Add, doublePrecision: true, out failure),
            Mnemonic.Subpd => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Sub, doublePrecision: true, out failure),
            Mnemonic.Mulpd => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Mul, doublePrecision: true, out failure),
            Mnemonic.Divpd => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Div, doublePrecision: true, out failure),
            Mnemonic.Minpd => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Min, doublePrecision: true, out failure),
            Mnemonic.Maxpd => ExecuteVectorArithLegacy(context, instruction, ScalarArithOp.Max, doublePrecision: true, out failure),
            Mnemonic.Xorps => ExecuteVectorBitwiseLegacy(context, instruction, VectorBitwiseOp.Xor, out failure),
            Mnemonic.Xorpd => ExecuteVectorBitwiseLegacy(context, instruction, VectorBitwiseOp.Xor, out failure),
            Mnemonic.Andps => ExecuteVectorBitwiseLegacy(context, instruction, VectorBitwiseOp.And, out failure),
            Mnemonic.Andpd => ExecuteVectorBitwiseLegacy(context, instruction, VectorBitwiseOp.And, out failure),
            Mnemonic.Andnps => ExecuteVectorBitwiseLegacy(context, instruction, VectorBitwiseOp.AndNot, out failure),
            Mnemonic.Andnpd => ExecuteVectorBitwiseLegacy(context, instruction, VectorBitwiseOp.AndNot, out failure),
            Mnemonic.Pxor => ExecuteVectorBitwiseLegacy(context, instruction, VectorBitwiseOp.Xor, out failure),
            Mnemonic.Pand => ExecuteVectorBitwiseLegacy(context, instruction, VectorBitwiseOp.And, out failure),
            Mnemonic.Por => ExecuteVectorBitwiseLegacy(context, instruction, VectorBitwiseOp.Or, out failure),
            Mnemonic.Pandn => ExecuteVectorBitwiseLegacy(context, instruction, VectorBitwiseOp.AndNot, out failure),
            Mnemonic.Paddb => ExecuteVectorPackedArithLegacy(context, instruction, PackedArithOp.Add, 8, out failure),
            Mnemonic.Paddw => ExecuteVectorPackedArithLegacy(context, instruction, PackedArithOp.Add, 16, out failure),
            Mnemonic.Paddd => ExecuteVectorPackedArithLegacy(context, instruction, PackedArithOp.Add, 32, out failure),
            Mnemonic.Paddq => ExecuteVectorPackedArithLegacy(context, instruction, PackedArithOp.Add, 64, out failure),
            Mnemonic.Psubb => ExecuteVectorPackedArithLegacy(context, instruction, PackedArithOp.Sub, 8, out failure),
            Mnemonic.Psubw => ExecuteVectorPackedArithLegacy(context, instruction, PackedArithOp.Sub, 16, out failure),
            Mnemonic.Psubd => ExecuteVectorPackedArithLegacy(context, instruction, PackedArithOp.Sub, 32, out failure),
            Mnemonic.Psubq => ExecuteVectorPackedArithLegacy(context, instruction, PackedArithOp.Sub, 64, out failure),
            Mnemonic.Pshufd => ExecuteVectorShuffleDwordsLegacy(context, instruction, out failure),
            Mnemonic.Pshuflw => ExecuteVectorShuffleLowWordsLegacy(context, instruction, out failure),
            Mnemonic.Pshufhw => ExecuteVectorShuffleHighWordsLegacy(context, instruction, out failure),
            Mnemonic.Punpcklbw => ExecuteVectorUnpackLegacy(context, instruction, UnpackHalf.Low, 8, out failure),
            Mnemonic.Punpckhbw => ExecuteVectorUnpackLegacy(context, instruction, UnpackHalf.High, 8, out failure),
            Mnemonic.Punpcklwd => ExecuteVectorUnpackLegacy(context, instruction, UnpackHalf.Low, 16, out failure),
            Mnemonic.Punpckhwd => ExecuteVectorUnpackLegacy(context, instruction, UnpackHalf.High, 16, out failure),
            Mnemonic.Punpckldq => ExecuteVectorUnpackLegacy(context, instruction, UnpackHalf.Low, 32, out failure),
            Mnemonic.Punpckhdq => ExecuteVectorUnpackLegacy(context, instruction, UnpackHalf.High, 32, out failure),
            Mnemonic.Punpcklqdq => ExecuteVectorUnpackLegacy(context, instruction, UnpackHalf.Low, 64, out failure),
            Mnemonic.Punpckhqdq => ExecuteVectorUnpackLegacy(context, instruction, UnpackHalf.High, 64, out failure),
            Mnemonic.Pshufb => ExecuteVectorShuffleBytesLegacy(context, instruction, out failure),
            Mnemonic.Pcmpeqb => ExecuteVectorCompareEqualBytesLegacy(context, instruction, out failure),
            Mnemonic.Pcmpeqd => ExecuteVectorCompareEqualDwordsLegacy(context, instruction, out failure),
            Mnemonic.Ucomiss => ExecuteFloatCompareLegacy(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Comiss => ExecuteFloatCompareLegacy(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Ucomisd => ExecuteFloatCompareLegacy(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Comisd => ExecuteFloatCompareLegacy(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Movmskps => ExecuteMoveMask(context, instruction, 4, out failure),
            Mnemonic.Movmskpd => ExecuteMoveMask(context, instruction, 8, out failure),
            Mnemonic.Pmovmskb => ExecuteMoveMask(context, instruction, 1, out failure),
            Mnemonic.Vzeroupper => ExecuteZeroUpper(context, out failure),
            Mnemonic.Vzeroall => ExecuteZeroAll(context, out failure),
            Mnemonic.Vmovups => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Vmovupd => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Vmovaps => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Vmovapd => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Vmovdqu => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Vmovdqa => ExecuteVectorMove(context, instruction, out failure),
            Mnemonic.Vmovhps => ExecuteVectorMoveHighPackedSingle(context, instruction, out failure),
            Mnemonic.Vmovd => ExecuteVectorMoveD(context, instruction, out failure),
            Mnemonic.Vmovq => ExecuteVectorMoveQ(context, instruction, out failure),
            Mnemonic.Vmovss => ExecuteMoveScalar(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Vmovsd => ExecuteMoveScalar(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Vcvtsi2ss => ExecuteConvertIntToScalar(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Vcvtsi2sd => ExecuteConvertIntToScalar(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Vcvttss2si => ExecuteConvertScalarToInt(context, instruction, doublePrecision: false, truncate: true, out failure),
            Mnemonic.Vcvttsd2si => ExecuteConvertScalarToInt(context, instruction, doublePrecision: true, truncate: true, out failure),
            Mnemonic.Vcvtss2si => ExecuteConvertScalarToInt(context, instruction, doublePrecision: false, truncate: false, out failure),
            Mnemonic.Vcvtsd2si => ExecuteConvertScalarToInt(context, instruction, doublePrecision: true, truncate: false, out failure),
            Mnemonic.Vaddss => ExecuteScalarArith(context, instruction, ScalarArithOp.Add, doublePrecision: false, out failure),
            Mnemonic.Vsubss => ExecuteScalarArith(context, instruction, ScalarArithOp.Sub, doublePrecision: false, out failure),
            Mnemonic.Vmulss => ExecuteScalarArith(context, instruction, ScalarArithOp.Mul, doublePrecision: false, out failure),
            Mnemonic.Vdivss => ExecuteScalarArith(context, instruction, ScalarArithOp.Div, doublePrecision: false, out failure),
            Mnemonic.Vaddsd => ExecuteScalarArith(context, instruction, ScalarArithOp.Add, doublePrecision: true, out failure),
            Mnemonic.Vsubsd => ExecuteScalarArith(context, instruction, ScalarArithOp.Sub, doublePrecision: true, out failure),
            Mnemonic.Vmulsd => ExecuteScalarArith(context, instruction, ScalarArithOp.Mul, doublePrecision: true, out failure),
            Mnemonic.Vdivsd => ExecuteScalarArith(context, instruction, ScalarArithOp.Div, doublePrecision: true, out failure),
            Mnemonic.Vaddps => ExecuteVectorArith(context, instruction, ScalarArithOp.Add, doublePrecision: false, out failure),
            Mnemonic.Vaddpd => ExecuteVectorArith(context, instruction, ScalarArithOp.Add, doublePrecision: true, out failure),
            Mnemonic.Vsubps => ExecuteVectorArith(context, instruction, ScalarArithOp.Sub, doublePrecision: false, out failure),
            Mnemonic.Vsubpd => ExecuteVectorArith(context, instruction, ScalarArithOp.Sub, doublePrecision: true, out failure),
            Mnemonic.Vmulps => ExecuteVectorArith(context, instruction, ScalarArithOp.Mul, doublePrecision: false, out failure),
            Mnemonic.Vmulpd => ExecuteVectorArith(context, instruction, ScalarArithOp.Mul, doublePrecision: true, out failure),
            Mnemonic.Vdivps => ExecuteVectorArith(context, instruction, ScalarArithOp.Div, doublePrecision: false, out failure),
            Mnemonic.Vdivpd => ExecuteVectorArith(context, instruction, ScalarArithOp.Div, doublePrecision: true, out failure),
            Mnemonic.Vminss => ExecuteScalarArith(context, instruction, ScalarArithOp.Min, doublePrecision: false, out failure),
            Mnemonic.Vminsd => ExecuteScalarArith(context, instruction, ScalarArithOp.Min, doublePrecision: true, out failure),
            Mnemonic.Vmaxss => ExecuteScalarArith(context, instruction, ScalarArithOp.Max, doublePrecision: false, out failure),
            Mnemonic.Vmaxsd => ExecuteScalarArith(context, instruction, ScalarArithOp.Max, doublePrecision: true, out failure),
            Mnemonic.Vminps => ExecuteVectorArith(context, instruction, ScalarArithOp.Min, doublePrecision: false, out failure),
            Mnemonic.Vminpd => ExecuteVectorArith(context, instruction, ScalarArithOp.Min, doublePrecision: true, out failure),
            Mnemonic.Vmaxps => ExecuteVectorArith(context, instruction, ScalarArithOp.Max, doublePrecision: false, out failure),
            Mnemonic.Vmaxpd => ExecuteVectorArith(context, instruction, ScalarArithOp.Max, doublePrecision: true, out failure),
            Mnemonic.Vpermilpd => ExecuteVectorPermuteLanes(context, instruction, elementBytes: 8, out failure),
            Mnemonic.Vpermilps => ExecuteVectorPermuteLanes(context, instruction, elementBytes: 4, out failure),
            Mnemonic.Vcmpsd => ExecuteScalarCompareToMask(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Vcmpss => ExecuteScalarCompareToMask(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Cmpsd => ExecuteScalarCompareToMask(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Cmpss => ExecuteScalarCompareToMask(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Vcmppd => ExecuteVectorCompareToMask(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Vcmpps => ExecuteVectorCompareToMask(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Cmppd => ExecuteVectorCompareToMask(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Cmpps => ExecuteVectorCompareToMask(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Vblendvpd => ExecuteVectorBlendVariable(context, instruction, elementBytes: 8, out failure),
            Mnemonic.Vblendvps => ExecuteVectorBlendVariable(context, instruction, elementBytes: 4, out failure),
            Mnemonic.Vpblendvb => ExecuteVectorBlendVariable(context, instruction, elementBytes: 1, out failure),
            Mnemonic.Blendvpd => ExecuteVectorBlendVariable(context, instruction, elementBytes: 8, out failure),
            Mnemonic.Blendvps => ExecuteVectorBlendVariable(context, instruction, elementBytes: 4, out failure),
            Mnemonic.Pblendvb => ExecuteVectorBlendVariable(context, instruction, elementBytes: 1, out failure),
            Mnemonic.Vextracti128 => ExecuteVectorExtract128(context, instruction, out failure),
            Mnemonic.Vextractf128 => ExecuteVectorExtract128(context, instruction, out failure),
            Mnemonic.Vinserti128 => ExecuteVectorInsert128(context, instruction, out failure),
            Mnemonic.Vinsertf128 => ExecuteVectorInsert128(context, instruction, out failure),
            Mnemonic.Vcvtss2sd => ExecuteConvertScalarPrecision(context, instruction, toDouble: true, out failure),
            Mnemonic.Vcvtsd2ss => ExecuteConvertScalarPrecision(context, instruction, toDouble: false, out failure),
            Mnemonic.Cvtss2sd => ExecuteConvertScalarPrecision(context, instruction, toDouble: true, out failure),
            Mnemonic.Cvtsd2ss => ExecuteConvertScalarPrecision(context, instruction, toDouble: false, out failure),
            Mnemonic.Vcvtpd2ps => ExecuteConvertPackedDoubleToSingle(context, instruction, out failure),
            Mnemonic.Cvtpd2ps => ExecuteConvertPackedDoubleToSingle(context, instruction, out failure),
            Mnemonic.Vcvtps2pd => ExecuteConvertPackedSingleToDouble(context, instruction, out failure),
            Mnemonic.Cvtps2pd => ExecuteConvertPackedSingleToDouble(context, instruction, out failure),
            Mnemonic.Vroundss => ExecuteRoundScalar(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Vroundsd => ExecuteRoundScalar(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Roundss => ExecuteRoundScalar(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Roundsd => ExecuteRoundScalar(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Vroundps => ExecuteRoundPacked(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Vroundpd => ExecuteRoundPacked(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Roundps => ExecuteRoundPacked(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Roundpd => ExecuteRoundPacked(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Vucomiss => ExecuteScalarCompare(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Vucomisd => ExecuteScalarCompare(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Vcomiss => ExecuteScalarCompare(context, instruction, doublePrecision: false, out failure),
            Mnemonic.Vcomisd => ExecuteScalarCompare(context, instruction, doublePrecision: true, out failure),
            Mnemonic.Vpextrq => ExecuteVectorExtractQ(context, instruction, out failure),
            Mnemonic.Vpextrd => ExecuteVectorExtractD(context, instruction, out failure),
            Mnemonic.Vpinsrd => ExecuteVectorInsertD(context, instruction, out failure),
            Mnemonic.Vpmovzxbw => ExecuteVectorMoveExtend(context, instruction, 8, 16, signed: false, out failure),
            Mnemonic.Vpmovzxbd => ExecuteVectorMoveExtend(context, instruction, 8, 32, signed: false, out failure),
            Mnemonic.Vpmovzxbq => ExecuteVectorMoveExtend(context, instruction, 8, 64, signed: false, out failure),
            Mnemonic.Vpmovzxwd => ExecuteVectorMoveExtend(context, instruction, 16, 32, signed: false, out failure),
            Mnemonic.Vpmovzxwq => ExecuteVectorMoveExtend(context, instruction, 16, 64, signed: false, out failure),
            Mnemonic.Vpmovzxdq => ExecuteVectorMoveExtend(context, instruction, 32, 64, signed: false, out failure),
            Mnemonic.Vpmovsxbw => ExecuteVectorMoveExtend(context, instruction, 8, 16, signed: true, out failure),
            Mnemonic.Vpmovsxbd => ExecuteVectorMoveExtend(context, instruction, 8, 32, signed: true, out failure),
            Mnemonic.Vpmovsxbq => ExecuteVectorMoveExtend(context, instruction, 8, 64, signed: true, out failure),
            Mnemonic.Vpmovsxwd => ExecuteVectorMoveExtend(context, instruction, 16, 32, signed: true, out failure),
            Mnemonic.Vpmovsxwq => ExecuteVectorMoveExtend(context, instruction, 16, 64, signed: true, out failure),
            Mnemonic.Vpmovsxdq => ExecuteVectorMoveExtend(context, instruction, 32, 64, signed: true, out failure),
            Mnemonic.Vbroadcastss => ExecuteVectorBroadcastScalarSingle(context, instruction, out failure),
            Mnemonic.Vbroadcastf128 => ExecuteVectorBroadcastF128(context, instruction, out failure),
            Mnemonic.Vpbroadcastq => ExecuteVectorBroadcastQ(context, instruction, out failure),
            Mnemonic.Vpbroadcastd => ExecuteVectorBroadcastD(context, instruction, out failure),
            Mnemonic.Vpsrlvd => ExecuteVectorVariableShift(context, instruction, VariableShiftOp.Srlv, 32, out failure),
            Mnemonic.Vpsrlvq => ExecuteVectorVariableShift(context, instruction, VariableShiftOp.Srlv, 64, out failure),
            Mnemonic.Vpsllvd => ExecuteVectorVariableShift(context, instruction, VariableShiftOp.Sllv, 32, out failure),
            Mnemonic.Vpsllvq => ExecuteVectorVariableShift(context, instruction, VariableShiftOp.Sllv, 64, out failure),
            Mnemonic.Vpsravd => ExecuteVectorVariableShift(context, instruction, VariableShiftOp.Srav, 32, out failure),
            Mnemonic.Vpsllw => ExecuteVectorPackedShift(context, instruction, PackedShiftOp.Sll, 16, out failure),
            Mnemonic.Vpslld => ExecuteVectorPackedShift(context, instruction, PackedShiftOp.Sll, 32, out failure),
            Mnemonic.Vpsllq => ExecuteVectorPackedShift(context, instruction, PackedShiftOp.Sll, 64, out failure),
            Mnemonic.Vpsrlw => ExecuteVectorPackedShift(context, instruction, PackedShiftOp.Srl, 16, out failure),
            Mnemonic.Vpsrld => ExecuteVectorPackedShift(context, instruction, PackedShiftOp.Srl, 32, out failure),
            Mnemonic.Vpsrlq => ExecuteVectorPackedShift(context, instruction, PackedShiftOp.Srl, 64, out failure),
            Mnemonic.Vpsraw => ExecuteVectorPackedShift(context, instruction, PackedShiftOp.Sra, 16, out failure),
            Mnemonic.Vpsrad => ExecuteVectorPackedShift(context, instruction, PackedShiftOp.Sra, 32, out failure),
            Mnemonic.Vxorps => ExecuteVectorXor(context, instruction, out failure),
            Mnemonic.Vxorpd => ExecuteVectorXor(context, instruction, out failure),
            Mnemonic.Vpxor => ExecuteVectorXor(context, instruction, out failure),
            Mnemonic.Vandps => ExecuteVectorBitwise(context, instruction, VectorBitwiseOp.And, out failure),
            Mnemonic.Vandpd => ExecuteVectorBitwise(context, instruction, VectorBitwiseOp.And, out failure),
            Mnemonic.Vorps => ExecuteVectorBitwise(context, instruction, VectorBitwiseOp.Or, out failure),
            Mnemonic.Vorpd => ExecuteVectorBitwise(context, instruction, VectorBitwiseOp.Or, out failure),
            Mnemonic.Vandnps => ExecuteVectorBitwise(context, instruction, VectorBitwiseOp.AndNot, out failure),
            Mnemonic.Vandnpd => ExecuteVectorBitwise(context, instruction, VectorBitwiseOp.AndNot, out failure),
            Mnemonic.Vpaddb => ExecuteVectorPackedArith(context, instruction, PackedArithOp.Add, 8, out failure),
            Mnemonic.Vpaddw => ExecuteVectorPackedArith(context, instruction, PackedArithOp.Add, 16, out failure),
            Mnemonic.Vpaddd => ExecuteVectorPackedArith(context, instruction, PackedArithOp.Add, 32, out failure),
            Mnemonic.Vpaddq => ExecuteVectorPackedArith(context, instruction, PackedArithOp.Add, 64, out failure),
            Mnemonic.Vpsubb => ExecuteVectorPackedArith(context, instruction, PackedArithOp.Sub, 8, out failure),
            Mnemonic.Vpsubw => ExecuteVectorPackedArith(context, instruction, PackedArithOp.Sub, 16, out failure),
            Mnemonic.Vpsubd => ExecuteVectorPackedArith(context, instruction, PackedArithOp.Sub, 32, out failure),
            Mnemonic.Vpsubq => ExecuteVectorPackedArith(context, instruction, PackedArithOp.Sub, 64, out failure),
            Mnemonic.Vpand => ExecuteVectorBitwise(context, instruction, VectorBitwiseOp.And, out failure),
            Mnemonic.Vpor => ExecuteVectorBitwise(context, instruction, VectorBitwiseOp.Or, out failure),
            Mnemonic.Vpandn => ExecuteVectorBitwise(context, instruction, VectorBitwiseOp.AndNot, out failure),
            Mnemonic.Vpshufb => ExecuteVectorShuffleBytes(context, instruction, out failure),
            Mnemonic.Vpshufd => ExecuteVectorShuffleDwords(context, instruction, out failure),
            Mnemonic.Vpblendw => ExecuteVectorBlendWords(context, instruction, out failure),
            Mnemonic.Vpblendd => ExecuteVectorBlendDwords(context, instruction, out failure),
            Mnemonic.Vpunpcklbw => ExecuteVectorUnpack(context, instruction, UnpackHalf.Low, 8, out failure),
            Mnemonic.Vpunpckhbw => ExecuteVectorUnpack(context, instruction, UnpackHalf.High, 8, out failure),
            Mnemonic.Vpunpcklwd => ExecuteVectorUnpack(context, instruction, UnpackHalf.Low, 16, out failure),
            Mnemonic.Vpunpckhwd => ExecuteVectorUnpack(context, instruction, UnpackHalf.High, 16, out failure),
            Mnemonic.Vpunpckldq => ExecuteVectorUnpack(context, instruction, UnpackHalf.Low, 32, out failure),
            Mnemonic.Vpunpckhdq => ExecuteVectorUnpack(context, instruction, UnpackHalf.High, 32, out failure),
            Mnemonic.Vpunpcklqdq => ExecuteVectorUnpack(context, instruction, UnpackHalf.Low, 64, out failure),
            Mnemonic.Vpunpckhqdq => ExecuteVectorUnpack(context, instruction, UnpackHalf.High, 64, out failure),
            Mnemonic.Vpcmpeqb => ExecuteVectorCompareEqualBytes(context, instruction, out failure),
            Mnemonic.Vpcmpeqd => ExecuteVectorCompareEqualDwords(context, instruction, out failure),
            Mnemonic.Vptest => ExecuteVectorTest(context, instruction, out failure),
            Mnemonic.Vpcmpistri => ExecuteVectorCompareImplicitStringIndex(context, instruction, out failure),
            Mnemonic.Lea => ExecuteLea(context, instruction, out failure),
            Mnemonic.Push => ExecutePush(context, instruction, out failure),
            Mnemonic.Pop => ExecutePop(context, instruction, out failure),
            Mnemonic.Inc => ExecuteUnary(context, instruction, UnaryOp.Inc, out failure),
            Mnemonic.Dec => ExecuteUnary(context, instruction, UnaryOp.Dec, out failure),
            Mnemonic.Neg => ExecuteUnary(context, instruction, UnaryOp.Neg, out failure),
            Mnemonic.Not => ExecuteUnary(context, instruction, UnaryOp.Not, out failure),
            Mnemonic.Bswap => ExecuteByteSwap(context, instruction, out failure),
            Mnemonic.Cbw => ExecuteSignExtendAccumulator(context, instruction.Mnemonic, out failure),
            Mnemonic.Cwde => ExecuteSignExtendAccumulator(context, instruction.Mnemonic, out failure),
            Mnemonic.Cdqe => ExecuteSignExtendAccumulator(context, instruction.Mnemonic, out failure),
            Mnemonic.Cwd => ExecuteSignExtendAccumulator(context, instruction.Mnemonic, out failure),
            Mnemonic.Cdq => ExecuteSignExtendAccumulator(context, instruction.Mnemonic, out failure),
            Mnemonic.Cqo => ExecuteSignExtendAccumulator(context, instruction.Mnemonic, out failure),
            Mnemonic.Div => ExecuteUnsignedDivide(context, instruction, out failure),
            Mnemonic.Idiv => ExecuteSignedDivide(context, instruction, out failure),
            Mnemonic.Mul => ExecuteUnsignedMultiply(context, instruction, out failure),
            Mnemonic.Imul => ExecuteSignedMultiply(context, instruction, out failure),
            Mnemonic.Add => ExecuteBinary(context, instruction, BinaryOp.Add, writeResult: true, out failure),
            Mnemonic.Sub => ExecuteBinary(context, instruction, BinaryOp.Sub, writeResult: true, out failure),
            Mnemonic.Adc => ExecuteAdc(context, instruction, out failure),
            Mnemonic.Sbb => ExecuteSbb(context, instruction, out failure),
            Mnemonic.Cmp => ExecuteBinary(context, instruction, BinaryOp.Sub, writeResult: false, out failure),
            Mnemonic.Cmpxchg => ExecuteCompareExchange(context, instruction, out failure),
            Mnemonic.Xadd => ExecuteExchangeAdd(context, instruction, out failure),
            Mnemonic.Xchg => ExecuteExchange(context, instruction, out failure),
            Mnemonic.Xor => ExecuteBinary(context, instruction, BinaryOp.Xor, writeResult: true, out failure),
            Mnemonic.And => ExecuteBinary(context, instruction, BinaryOp.And, writeResult: true, out failure),
            Mnemonic.Or => ExecuteBinary(context, instruction, BinaryOp.Or, writeResult: true, out failure),
            Mnemonic.Test => ExecuteBinary(context, instruction, BinaryOp.And, writeResult: false, out failure),
            Mnemonic.Shl => ExecuteShift(context, instruction, ShiftOp.Shl, out failure),
            Mnemonic.Sal => ExecuteShift(context, instruction, ShiftOp.Shl, out failure),
            Mnemonic.Shr => ExecuteShift(context, instruction, ShiftOp.Shr, out failure),
            Mnemonic.Sar => ExecuteShift(context, instruction, ShiftOp.Sar, out failure),
            Mnemonic.Shlx => ExecuteShiftx(context, instruction, ShiftxOp.Shlx, out failure),
            Mnemonic.Shrx => ExecuteShiftx(context, instruction, ShiftxOp.Shrx, out failure),
            Mnemonic.Sarx => ExecuteShiftx(context, instruction, ShiftxOp.Sarx, out failure),
            Mnemonic.Bextr => ExecuteBitFieldExtract(context, instruction, out failure),
            Mnemonic.Rorx => ExecuteRotateRightNoFlags(context, instruction, out failure),
            Mnemonic.Bzhi => ExecuteZeroHighBits(context, instruction, out failure),
            Mnemonic.Bt => ExecuteBitTest(context, instruction, out failure),
            Mnemonic.Popcnt => ExecutePopCount(context, instruction, out failure),
            Mnemonic.Lzcnt => ExecuteLeadingZeroCount(context, instruction, out failure),
            _ => Unsupported(instruction, out failure),
        };
    }

    // -------------------------------------------------------------------------
    // Legacy SSE/SSE2 compatibility layer.
    //
    // VEX instructions are normally three-operand. Legacy SSE is two-operand:
    //   addps xmm0,xmm1  == xmm0 = xmm0 + xmm1
    // The original backend already implements the arithmetic kernels for VEX;
    // these small adapters provide the correct legacy operand semantics.
    // -------------------------------------------------------------------------

    private static bool ExecuteVectorMoveHighPackedSingleLegacy(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.OpCount != 2 || instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var dst, out _))
        { failure = InterpreterFailure.Unsupported($"unsupported MOVHPS: {instruction}"); return false; }
        if (instruction.GetOpKind(1) == OpKind.Memory)
        {
            if (!TryGetMemoryAddress(context, instruction, out var address)) { failure=InterpreterFailure.Unsupported("unsupported MOVHPS address"); return false; }
            Span<byte> tmp=stackalloc byte[8]; if(!TryReadInterpreterMemory(context,address,tmp)){failure=InterpreterFailure.MemoryRead(address,8);return false;}
            context.GetXmmRegister(dst,out var low,out var high);
            high=BinaryPrimitives.ReadUInt64LittleEndian(tmp);
            context.SetXmmRegister(dst,low,high); context.ClearYmmUpper(dst); failure=default; return true;
        }
        if(!TryGetVectorRegisterInfo(instruction.GetOpRegister(1),out var src,out _)){failure=InterpreterFailure.Unsupported($"unsupported MOVHPS source: {instruction}");return false;}
        context.GetXmmRegister(src,out _,out var srcHigh); context.GetXmmRegister(dst,out var dstLow,out _); context.SetXmmRegister(dst,dstLow,srcHigh); context.ClearYmmUpper(dst); failure=default; return true;
    }

    private static bool ExecuteVectorMoveLowPackedSingleLegacy(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.OpCount != 2 || instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var dst, out _))
        {
            if (instruction.OpCount == 2 && instruction.GetOpKind(0) == OpKind.Memory && instruction.GetOpKind(1) == OpKind.Register && TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var storeSrc, out _))
            {
                if(!TryGetMemoryAddress(context,instruction,out var address)){failure=InterpreterFailure.Unsupported("unsupported MOVLPS store address");return false;}
                context.GetXmmRegister(storeSrc,out var low,out _); Span<byte> tmp=stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(tmp,low); if(!TryWriteInterpreterMemory(context,address,tmp)){failure=InterpreterFailure.MemoryWrite(address,8);return false;} failure=default; return true;
            }
            failure=InterpreterFailure.Unsupported($"unsupported MOVLPS: {instruction}"); return false;
        }
        if(instruction.GetOpKind(1)==OpKind.Memory)
        {
            if(!TryGetMemoryAddress(context,instruction,out var address)){failure=InterpreterFailure.Unsupported("unsupported MOVLPS load address");return false;}
            Span<byte> tmp=stackalloc byte[8]; if(!TryReadInterpreterMemory(context,address,tmp)){failure=InterpreterFailure.MemoryRead(address,8);return false;}
            context.GetXmmRegister(dst,out _,out var high); context.SetXmmRegister(dst,BinaryPrimitives.ReadUInt64LittleEndian(tmp),high); context.ClearYmmUpper(dst); failure=default; return true;
        }
        if(!TryGetVectorRegisterInfo(instruction.GetOpRegister(1),out var src,out _)){failure=InterpreterFailure.Unsupported($"unsupported MOVLPS source: {instruction}");return false;}
        context.GetXmmRegister(src,out var srcLow,out _); context.GetXmmRegister(dst,out _,out var dstHigh); context.SetXmmRegister(dst,srcLow,dstHigh); context.ClearYmmUpper(dst); failure=default; return true;
    }

    private static bool ExecuteMoveScalarLegacy(CpuContext context, in Instruction instruction, bool doublePrecision, out InterpreterFailure failure)
    {
        var bytes = doublePrecision ? 8 : 4;
        if (instruction.GetOpKind(0) == OpKind.Memory)
        {
            if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var src, out _))
            {
                failure = InterpreterFailure.Unsupported($"unsupported scalar store: {instruction}");
                return false;
            }
            if (!TryGetMemoryAddress(context, instruction, out var address))
            {
                failure = InterpreterFailure.Unsupported("unsupported scalar store address");
                return false;
            }
            Span<byte> tmp = stackalloc byte[16];
            GetVectorRegister(context, src, 16, tmp);
            if (!TryWriteInterpreterMemory(context, address, tmp[..bytes]))
            {
                failure = InterpreterFailure.MemoryWrite(address, bytes);
                return false;
            }
            failure = default;
            return true;
        }

        if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var dst, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported scalar destination: {instruction}");
            return false;
        }

        if (instruction.GetOpKind(1) == OpKind.Memory)
        {
            if (!TryGetMemoryAddress(context, instruction, out var address))
            {
                failure = InterpreterFailure.Unsupported("unsupported scalar load address");
                return false;
            }
            Span<byte> tmp = stackalloc byte[16];
            if (!TryReadInterpreterMemory(context, address, tmp[..bytes]))
            {
                failure = InterpreterFailure.MemoryRead(address, bytes);
                return false;
            }
            context.GetXmmRegister(dst, out var oldLow, out var oldHigh);
            if (doublePrecision)
                context.SetXmmRegister(dst, BinaryPrimitives.ReadUInt64LittleEndian(tmp), oldHigh);
            else
                context.SetXmmRegister(dst, (oldLow & 0xFFFFFFFF00000000UL) | BinaryPrimitives.ReadUInt32LittleEndian(tmp), oldHigh);
            context.ClearYmmUpper(dst);
            failure = default;
            return true;
        }

        if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var srcReg, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported scalar register source: {instruction}");
            return false;
        }
        context.GetXmmRegister(srcReg, out var srcLow, out var srcHigh);
        context.GetXmmRegister(dst, out var dstLow, out var dstHigh);
        if (doublePrecision)
            context.SetXmmRegister(dst, srcLow, dstHigh);
        else
            context.SetXmmRegister(dst, (dstLow & 0xFFFFFFFF00000000UL) | (srcLow & 0xFFFFFFFFUL), dstHigh);
        context.ClearYmmUpper(dst);
        failure = default;
        return true;
    }

    private static bool ExecuteScalarArithLegacy(CpuContext context, in Instruction instruction, ScalarArithOp op, bool doublePrecision, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var dst, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported legacy scalar arithmetic: {instruction}");
            return false;
        }
        context.GetXmmRegister(dst, out var low, out var high);
        double d;
        float f;
        if (doublePrecision)
        {
            if (!TryReadScalarDouble(context, instruction, 1, out d, out failure)) return false;
            var a = BitConverter.UInt64BitsToDouble(low);
            var r = op switch { ScalarArithOp.Add => a+d, ScalarArithOp.Sub => a-d, ScalarArithOp.Mul => a*d, ScalarArithOp.Div => a/d, ScalarArithOp.Min => EvaluateMin(a,d), ScalarArithOp.Max => EvaluateMax(a,d), _ => a };
            context.SetXmmRegister(dst, BitConverter.DoubleToUInt64Bits(r), high);
        }
        else
        {
            if (!TryReadScalarSingle(context, instruction, 1, out f, out failure)) return false;
            var a = BitConverter.UInt32BitsToSingle((uint)low);
            var r = op switch { ScalarArithOp.Add => a+f, ScalarArithOp.Sub => a-f, ScalarArithOp.Mul => a*f, ScalarArithOp.Div => a/f, ScalarArithOp.Min => EvaluateMinSingle(a,f), ScalarArithOp.Max => EvaluateMaxSingle(a,f), _ => a };
            context.SetXmmRegister(dst, (low & 0xFFFFFFFF00000000UL) | BitConverter.SingleToUInt32Bits(r), high);
        }
        context.ClearYmmUpper(dst);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorArithLegacy(CpuContext context, in Instruction instruction, ScalarArithOp op, bool doublePrecision, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var dst, out var size) ||
            instruction.OpCount != 2)
        {
            failure = InterpreterFailure.Unsupported($"unsupported legacy packed arithmetic: {instruction}");
            return false;
        }
        Span<byte> left = stackalloc byte[32];
        Span<byte> right = stackalloc byte[32];
        GetVectorRegister(context, dst, size, left[..size]);
        if (!TryReadVectorOperand(context, instruction, 1, size, right[..size], out failure)) return false;
        var elem = doublePrecision ? 8 : 4;
        Span<byte> result = stackalloc byte[32];
        for (var i=0; i<size; i+=elem)
        {
            if (doublePrecision)
            {
                var a=BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(left[i..]));
                var b=BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(right[i..]));
                var r=op switch { ScalarArithOp.Add=>a+b, ScalarArithOp.Sub=>a-b, ScalarArithOp.Mul=>a*b, ScalarArithOp.Div=>a/b, ScalarArithOp.Min=>EvaluateMin(a,b), ScalarArithOp.Max=>EvaluateMax(a,b), _=>a };
                BinaryPrimitives.WriteUInt64LittleEndian(result[i..], BitConverter.DoubleToUInt64Bits(r));
            }
            else
            {
                var a=BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(left[i..]));
                var b=BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(right[i..]));
                var r=op switch { ScalarArithOp.Add=>a+b, ScalarArithOp.Sub=>a-b, ScalarArithOp.Mul=>a*b, ScalarArithOp.Div=>a/b, ScalarArithOp.Min=>EvaluateMinSingle(a,b), ScalarArithOp.Max=>EvaluateMaxSingle(a,b), _=>a };
                BinaryPrimitives.WriteUInt32LittleEndian(result[i..], BitConverter.SingleToUInt32Bits(r));
            }
        }
        SetVectorRegister(context,dst,result[..size]);
        failure=default;
        return true;
    }

    private static bool ExecuteVectorBitwiseLegacy(CpuContext context, in Instruction instruction, VectorBitwiseOp op, out InterpreterFailure failure)
    {
        if (instruction.OpCount != 2 || instruction.GetOpKind(0) != OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var dst, out var size))
        { failure=InterpreterFailure.Unsupported($"unsupported legacy bitwise: {instruction}"); return false; }
        Span<byte> a=stackalloc byte[32]; Span<byte> b=stackalloc byte[32];
        GetVectorRegister(context,dst,size,a[..size]);
        if(!TryReadVectorOperand(context,instruction,1,size,b[..size],out failure)) return false;
        for(int i=0;i<size;i++) a[i]=op switch { VectorBitwiseOp.And=>(byte)(a[i]&b[i]), VectorBitwiseOp.Or=>(byte)(a[i]|b[i]), VectorBitwiseOp.Xor=>(byte)(a[i]^b[i]), VectorBitwiseOp.AndNot=>(byte)(~a[i]&b[i]), _=>a[i] };
        SetVectorRegister(context,dst,a[..size]); failure=default; return true;
    }

    private static bool ExecuteVectorPackedArithLegacy(CpuContext context, in Instruction instruction, PackedArithOp op, int elementBits, out InterpreterFailure failure)
    {
        if (instruction.OpCount != 2 || instruction.GetOpKind(0) != OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var dst, out var size))
        { failure=InterpreterFailure.Unsupported($"unsupported legacy packed integer arithmetic: {instruction}"); return false; }
        Span<byte> a=stackalloc byte[32]; Span<byte> b=stackalloc byte[32]; Span<byte> r=stackalloc byte[32];
        GetVectorRegister(context,dst,size,a[..size]); if(!TryReadVectorOperand(context,instruction,1,size,b[..size],out failure)) return false;
        int n=elementBits/8;
        for(int i=0;i<size;i+=n) { ulong x=n switch {1=>a[i],2=>BinaryPrimitives.ReadUInt16LittleEndian(a[i..]),4=>BinaryPrimitives.ReadUInt32LittleEndian(a[i..]),_=>BinaryPrimitives.ReadUInt64LittleEndian(a[i..])}; ulong y=n switch {1=>b[i],2=>BinaryPrimitives.ReadUInt16LittleEndian(b[i..]),4=>BinaryPrimitives.ReadUInt32LittleEndian(b[i..]),_=>BinaryPrimitives.ReadUInt64LittleEndian(b[i..])}; ulong z=op==PackedArithOp.Add?x+y:x-y; if(n==1)r[i]=(byte)z; else if(n==2)BinaryPrimitives.WriteUInt16LittleEndian(r[i..],(ushort)z); else if(n==4)BinaryPrimitives.WriteUInt32LittleEndian(r[i..],(uint)z); else BinaryPrimitives.WriteUInt64LittleEndian(r[i..],z); }
        SetVectorRegister(context,dst,r[..size]); failure=default; return true;
    }

    private static bool ExecuteVectorShuffleDwordsLegacy(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=3 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0),out var dst,out var size)) { failure=InterpreterFailure.Unsupported($"unsupported PSHUFD: {instruction}"); return false; }
        Span<byte> src=stackalloc byte[16]; if(!TryReadVectorOperand(context,instruction,1,16,src,out failure)) return false;
        byte imm=(byte)instruction.GetImmediate(2); Span<byte> r=stackalloc byte[16];
        for(int i=0;i<4;i++) BinaryPrimitives.WriteUInt32LittleEndian(r[(i*4)..],BinaryPrimitives.ReadUInt32LittleEndian(src[((imm>>(i*2)&3)*4)..]));
        SetVectorRegister(context,dst,r); failure=default; return true;
    }

    private static bool ExecuteVectorShuffleLowWordsLegacy(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=3 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0),out var dst,out _)) { failure=InterpreterFailure.Unsupported($"unsupported PSHUFLW: {instruction}"); return false; }
        Span<byte> v=stackalloc byte[16]; if(!TryReadVectorOperand(context,instruction,1,16,v,out failure)) return false; byte imm=(byte)instruction.GetImmediate(2); Span<byte> r=stackalloc byte[16]; v.CopyTo(r);
        for(int i=0;i<4;i++) BinaryPrimitives.WriteUInt16LittleEndian(r[(i*2)..],BinaryPrimitives.ReadUInt16LittleEndian(v[(((imm>>(i*2))&3)*2)..]));
        SetVectorRegister(context,dst,r); failure=default; return true;
    }

    private static bool ExecuteVectorShuffleHighWordsLegacy(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=3 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0),out var dst,out _)) { failure=InterpreterFailure.Unsupported($"unsupported PSHUFHW: {instruction}"); return false; }
        Span<byte> v=stackalloc byte[16]; if(!TryReadVectorOperand(context,instruction,1,16,v,out failure)) return false; byte imm=(byte)instruction.GetImmediate(2); Span<byte> r=stackalloc byte[16]; v.CopyTo(r);
        for(int i=0;i<4;i++) BinaryPrimitives.WriteUInt16LittleEndian(r[(8+i*2)..],BinaryPrimitives.ReadUInt16LittleEndian(v[(8+(((imm>>(i*2))&3)*2))..]));
        SetVectorRegister(context,dst,r); failure=default; return true;
    }

    private static bool ExecuteVectorUnpackLegacy(CpuContext context, in Instruction instruction, UnpackHalf half, int elementBits, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=2 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0),out var dst,out var size)) { failure=InterpreterFailure.Unsupported($"unsupported legacy unpack: {instruction}"); return false; }
        Span<byte> left=stackalloc byte[16]; Span<byte> right=stackalloc byte[16]; Span<byte> r=stackalloc byte[16]; GetVectorRegister(context,dst,16,left); if(!TryReadVectorOperand(context,instruction,1,16,right,out failure)) return false;
        int es=elementBits/8, halfCount=8/es, baseIndex=half==UnpackHalf.High?halfCount:0;
        for(int i=0;i<halfCount;i++){ int src=(baseIndex+i)*es; left.AsSpan(src,es).CopyTo(r.AsSpan(2*i*es,es)); right.AsSpan(src,es).CopyTo(r.AsSpan((2*i+1)*es,es)); }
        SetVectorRegister(context,dst,r); failure=default; return true;
    }

    private static bool ExecuteVectorShuffleBytesLegacy(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=2 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0),out var dst,out _)) { failure=InterpreterFailure.Unsupported($"unsupported PSHUFB: {instruction}"); return false; }
        Span<byte> src=stackalloc byte[16]; Span<byte> ctl=stackalloc byte[16]; Span<byte> r=stackalloc byte[16]; GetVectorRegister(context,dst,16,src); if(!TryReadVectorOperand(context,instruction,1,16,ctl,out failure)) return false;
        for(int lane=0;lane<16;lane+=16) for(int i=0;i<16;i++){ byte c=ctl[lane+i]; r[lane+i]=(c&0x80)!=0?(byte)0:src[lane+(c&0x0F)]; }
        SetVectorRegister(context,dst,r); failure=default; return true;
    }

    private static bool ExecuteVectorCompareEqualBytesLegacy(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=2 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0),out var dst,out var size)) { failure=InterpreterFailure.Unsupported($"unsupported PCMPEQB: {instruction}"); return false; }
        Span<byte> a=stackalloc byte[32]; Span<byte> b=stackalloc byte[32]; GetVectorRegister(context,dst,size,a[..size]); if(!TryReadVectorOperand(context,instruction,1,size,b[..size],out failure)) return false; for(int i=0;i<size;i++) a[i]=(byte)(a[i]==b[i]?0xFF:0); SetVectorRegister(context,dst,a[..size]); failure=default; return true;
    }

    private static bool ExecuteVectorCompareEqualDwordsLegacy(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=2 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0),out var dst,out var size)) { failure=InterpreterFailure.Unsupported($"unsupported PCMPEQD: {instruction}"); return false; }
        Span<byte> a=stackalloc byte[32]; Span<byte> b=stackalloc byte[32]; GetVectorRegister(context,dst,size,a[..size]); if(!TryReadVectorOperand(context,instruction,1,size,b[..size],out failure)) return false; for(int i=0;i<size;i+=4){ uint x=BinaryPrimitives.ReadUInt32LittleEndian(a[i..]); uint y=BinaryPrimitives.ReadUInt32LittleEndian(b[i..]); BinaryPrimitives.WriteUInt32LittleEndian(a[i..],x==y?uint.MaxValue:0); } SetVectorRegister(context,dst,a[..size]); failure=default; return true;
    }

    private static bool ExecuteFloatCompareLegacy(CpuContext context, in Instruction instruction, bool doublePrecision, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=2 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0),out var leftIndex,out _)) { failure=InterpreterFailure.Unsupported($"unsupported float compare: {instruction}"); return false; }
        context.GetXmmRegister(leftIndex,out var low,out _); bool unordered; int cmp;
        if(doublePrecision){ if(!TryReadScalarDouble(context,instruction,1,out var b,out failure)) return false; var a=BitConverter.UInt64BitsToDouble(low); unordered=double.IsNaN(a)||double.IsNaN(b); cmp=unordered?0:a.CompareTo(b); }
        else { if(!TryReadScalarSingle(context,instruction,1,out var b,out failure)) return false; var a=BitConverter.UInt32BitsToSingle((uint)low); unordered=float.IsNaN(a)||float.IsNaN(b); cmp=unordered?0:a.CompareTo(b); }
        const ulong CF=1UL, PF=1UL<<2, ZF=1UL<<6, SF=1UL<<7, OF=1UL<<11, AF=1UL<<4;
        context.Rflags &= ~(CF|PF|ZF|SF|OF|AF); if(unordered) context.Rflags|=CF|PF|ZF; else if(cmp<0) context.Rflags|=CF; else if(cmp==0) context.Rflags|=ZF; failure=default; return true;
    }

    private static bool ExecuteMoveMask(CpuContext context, in Instruction instruction, int elementBytes, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=2 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(1),out var src,out var size)) { failure=InterpreterFailure.Unsupported($"unsupported movemask: {instruction}"); return false; }
        context.GetXmmRegister(src,out var low,out var high); ulong mask=0; if(elementBytes==1){ for(int i=0;i<16;i++){ ulong q=i<8?low:high; if(((q>>((i&7)*8+7))&1)!=0) mask|=1UL<<i; }} else if(elementBytes==4){ for(int i=0;i<4;i++){ ulong q=i<2?low:high; int lane=i&1; if(((q>>(lane*32+31))&1)!=0) mask|=1UL<<i; }} else { for(int i=0;i<2;i++){ ulong q=i==0?low:high; if((q>>63)!=0) mask|=1UL<<i; }}
        return TryWriteOperand(context,instruction,0,mask,GetOperandBitSize(instruction,0),out failure);
    }

    private static bool ExecuteZeroUpper(CpuContext context, out InterpreterFailure failure)
    {
        for (var i = 0; i < 16; i++) context.ClearYmmUpper(i);
        failure = default;
        return true;
    }

    private static bool ExecuteZeroAll(CpuContext context, out InterpreterFailure failure)
    {
        for (var i = 0; i < 16; i++) context.SetYmmRegister(i, 0, 0, 0, 0);
        failure = default;
        return true;
    }

    private static bool ExecuteMovHlps(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.OpCount != 2 || instruction.GetOpKind(0) != OpKind.Register || instruction.GetOpKind(1) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var dst, out _) ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var src, out _))
        { failure=InterpreterFailure.Unsupported($"unsupported MOVHLPS: {instruction}"); return false; }
        context.GetXmmRegister(src,out _,out var srcHigh); context.GetXmmRegister(dst,out var dstLow,out _); context.SetXmmRegister(dst,srcHigh,dstLow); context.ClearYmmUpper(dst); failure=default; return true;
    }

    private static bool ExecuteMovLhps(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.OpCount != 2 || instruction.GetOpKind(0) != OpKind.Register || instruction.GetOpKind(1) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var dst, out _) ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var src, out _))
        { failure=InterpreterFailure.Unsupported($"unsupported MOVLHPS: {instruction}"); return false; }
        context.GetXmmRegister(src,out var srcLow,out _); context.GetXmmRegister(dst,out var dstLow,out _); context.SetXmmRegister(dst,dstLow,srcLow); context.ClearYmmUpper(dst); failure=default; return true;
    }

    private static bool ExecuteSqrt(CpuContext context, in Instruction instruction, bool doublePrecision, bool packed, out InterpreterFailure failure)
    {
        if (instruction.OpCount != 2 || instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var dst, out var size))
        { failure=InterpreterFailure.Unsupported($"unsupported SQRT: {instruction}"); return false; }
        var elem=doublePrecision?8:4;
        if(!packed) size=elem;
        Span<byte> src=stackalloc byte[32]; Span<byte> r=stackalloc byte[32];
        if(!TryReadVectorOperand(context,instruction,1,size,src[..size],out failure)) return false;
        for(int i=0;i<size;i+=elem){ if(doublePrecision){var v=BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(src[i..])); BinaryPrimitives.WriteUInt64LittleEndian(r[i..],BitConverter.DoubleToUInt64Bits(Math.Sqrt(v)));} else {var v=BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(src[i..])); BinaryPrimitives.WriteUInt32LittleEndian(r[i..],BitConverter.SingleToUInt32Bits(MathF.Sqrt(v)));}}
        if(!packed){ context.GetXmmRegister(dst,out var oldLow,out var oldHigh); if(doublePrecision) context.SetXmmRegister(dst,BinaryPrimitives.ReadUInt64LittleEndian(r),oldHigh); else context.SetXmmRegister(dst,(oldLow&0xFFFFFFFF00000000UL)|BinaryPrimitives.ReadUInt32LittleEndian(r),oldHigh); context.ClearYmmUpper(dst); }
        else SetVectorRegister(context,dst,r[..size]);
        failure=default; return true;
    }

    private static bool ExecuteReciprocalSqrt(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
        => ExecuteFloatUnaryPacked(context,instruction,reciprocalSqrt:true,out failure);
    private static bool ExecuteReciprocal(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
        => ExecuteFloatUnaryPacked(context,instruction,reciprocalSqrt:false,out failure);
    private static bool ExecuteReciprocalSqrtScalar(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
        => ExecuteFloatUnaryScalar(context,instruction,reciprocalSqrt:true,out failure);
    private static bool ExecuteReciprocalScalar(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
        => ExecuteFloatUnaryScalar(context,instruction,reciprocalSqrt:false,out failure);

    private static bool ExecuteFloatUnaryPacked(CpuContext context, in Instruction instruction, bool reciprocalSqrt, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=2 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0),out var dst,out var size)){failure=InterpreterFailure.Unsupported($"unsupported packed reciprocal: {instruction}");return false;}
        Span<byte> src=stackalloc byte[32]; Span<byte> r=stackalloc byte[32]; if(!TryReadVectorOperand(context,instruction,1,size,src[..size],out failure))return false;
        for(int i=0;i<size;i+=4){float v=BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(src[i..])); float x=reciprocalSqrt?1f/MathF.Sqrt(v):1f/v; BinaryPrimitives.WriteUInt32LittleEndian(r[i..],BitConverter.SingleToUInt32Bits(x));}
        SetVectorRegister(context,dst,r[..size]); failure=default;return true;
    }

    private static bool ExecuteFloatUnaryScalar(CpuContext context, in Instruction instruction, bool reciprocalSqrt, out InterpreterFailure failure)
    {
        if(instruction.OpCount!=2 || instruction.GetOpKind(0)!=OpKind.Register || !TryGetVectorRegisterInfo(instruction.GetOpRegister(0),out var dst,out _)){failure=InterpreterFailure.Unsupported($"unsupported scalar reciprocal: {instruction}");return false;}
        context.GetXmmRegister(dst,out var low,out var high); if(!TryReadScalarSingle(context,instruction,1,out var v,out failure))return false; float x=reciprocalSqrt?1f/MathF.Sqrt(v):1f/v; context.SetXmmRegister(dst,(low&0xFFFFFFFF00000000UL)|BitConverter.SingleToUInt32Bits(x),high); context.ClearYmmUpper(dst); failure=default;return true;
    }

    private static bool ExecuteMove(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (!TryReadOperand(context, instruction, 1, GetOperandBitSize(instruction, 0), out var value, out failure))
        {
            return false;
        }
        return TryWriteOperand(context, instruction, 0, value, GetOperandBitSize(instruction, 0), out failure);
    }

    private static bool ExecuteMovzx(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (!TryReadOperand(context, instruction, 1, GetOperandBitSize(instruction, 1), out var value, out failure))
        {
            return false;
        }
        return TryWriteOperand(context, instruction, 0, value, GetOperandBitSize(instruction, 0), out failure);
    }

    private static bool ExecuteMovsxd(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (!TryReadOperand(context, instruction, 1, GetOperandBitSize(instruction, 1), out var value, out failure))
        {
            return false;
        }

        var signed = SignExtend(value, 32);
        return TryWriteOperand(context, instruction, 0, signed, GetOperandBitSize(instruction, 0), out failure);
    }

    private static bool ExecuteMovsx(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var sourceBits = GetOperandBitSize(instruction, 1);
        if (!TryReadOperand(context, instruction, 1, sourceBits, out var value, out failure))
        {
            return false;
        }

        var signed = SignExtend(value, sourceBits);
        return TryWriteOperand(context, instruction, 0, signed, GetOperandBitSize(instruction, 0), out failure);
    }

    private static ulong SignExtend(ulong value, int bits)
    {
        if (bits >= 64)
        {
            return value;
        }

        var shift = 64 - bits;
        return (ulong)((long)(value << shift) >> shift);
    }

    private static bool ExecuteConditionalMove(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (!X64Flags.TryEvaluateCondition(instruction.Mnemonic, context.Rflags, out var isTaken))
        {
            failure = InterpreterFailure.Unsupported($"unsupported conditional move {instruction.Mnemonic}");
            return false;
        }
        if (!isTaken)
        {
            failure = default;
            return true;
        }

        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 1, bits, out var value, out failure))
        {
            return false;
        }

        return TryWriteOperand(context, instruction, 0, value, bits, out failure);
    }

    private static bool ExecuteSetCondition(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (!X64Flags.TryEvaluateCondition(instruction.Mnemonic, context.Rflags, out var isTaken))
        {
            failure = InterpreterFailure.Unsupported($"unsupported setcc {instruction.Mnemonic}");
            return false;
        }

        return TryWriteOperand(context, instruction, 0, isTaken ? 1UL : 0, 8, out failure);
    }

    private static bool ExecuteCompareExchange(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 1);
        if (bits is not (8 or 16 or 32 or 64))
        {
            failure = InterpreterFailure.Unsupported($"unsupported CMPXCHG size {bits}: {instruction}");
            return false;
        }

        if (!TryReadOperand(context, instruction, 0, bits, out var destination, out failure) ||
            !TryReadOperand(context, instruction, 1, bits, out var source, out failure))
        {
            return false;
        }

        var accumulator = context[CpuRegister.Rax] & X64Flags.Mask(bits);
        var result = (accumulator - destination) & X64Flags.Mask(bits);
        context.Rflags = X64Flags.UpdateSub(context.Rflags, accumulator, destination, result, bits);

        if (result == 0)
        {
            return TryWriteOperand(context, instruction, 0, source, bits, out failure);
        }

        return TryWriteAccumulator(context, destination, bits, out failure);
    }

    private static bool ExecuteExchangeAdd(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 1);
        if (bits is not (8 or 16 or 32 or 64))
        {
            failure = InterpreterFailure.Unsupported($"unsupported XADD size {bits}: {instruction}");
            return false;
        }

        if (!TryReadOperand(context, instruction, 0, bits, out var destination, out failure) ||
            !TryReadOperand(context, instruction, 1, bits, out var source, out failure))
        {
            return false;
        }

        var result = (destination + source) & X64Flags.Mask(bits);
        context.Rflags = X64Flags.UpdateAdd(context.Rflags, destination, source, result, bits);

        if (!TryWriteOperand(context, instruction, 0, result, bits, out failure))
        {
            return false;
        }

        return TryWriteOperand(context, instruction, 1, destination, bits, out failure);
    }

    private static bool ExecuteExchange(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 1);
        if (bits is not (8 or 16 or 32 or 64))
        {
            failure = InterpreterFailure.Unsupported($"unsupported XCHG size {bits}: {instruction}");
            return false;
        }

        if (!TryReadOperand(context, instruction, 0, bits, out var first, out failure) ||
            !TryReadOperand(context, instruction, 1, bits, out var second, out failure))
        {
            return false;
        }

        if (!TryWriteOperand(context, instruction, 0, second, bits, out failure))
        {
            return false;
        }

        return TryWriteOperand(context, instruction, 1, first, bits, out failure);
    }

    private static bool ExecutePopCount(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 1);
        if (bits is not (16 or 32 or 64))
        {
            failure = InterpreterFailure.Unsupported($"unsupported POPCNT size {bits}: {instruction}");
            return false;
        }

        if (!TryReadOperand(context, instruction, 1, bits, out var source, out failure))
        {
            return false;
        }

        source &= X64Flags.Mask(bits);
        var result = (ulong)BitOperations.PopCount(source);
        context.Rflags = X64Flags.UpdatePopCount(context.Rflags, source);
        return TryWriteOperand(context, instruction, 0, result, GetOperandBitSize(instruction, 0), out failure);
    }

    private static bool ExecuteLeadingZeroCount(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 1);
        if (bits is not (16 or 32 or 64))
        {
            failure = InterpreterFailure.Unsupported($"unsupported LZCNT size {bits}: {instruction}");
            return false;
        }

        if (!TryReadOperand(context, instruction, 1, bits, out var source, out failure))
        {
            return false;
        }

        source &= X64Flags.Mask(bits);
        var result = bits switch
        {
            16 => (ulong)(BitOperations.LeadingZeroCount((uint)source) - 16),
            32 => (ulong)BitOperations.LeadingZeroCount((uint)source),
            64 => (ulong)BitOperations.LeadingZeroCount(source),
            _ => 0UL,
        };

        context.Rflags = X64Flags.UpdateBitScanCount(context.Rflags, source, result);
        return TryWriteOperand(context, instruction, 0, result, GetOperandBitSize(instruction, 0), out failure);
    }

    private static bool ExecuteSignExtendAccumulator(CpuContext context, Mnemonic mnemonic, out InterpreterFailure failure)
    {
        switch (mnemonic)
        {
            case Mnemonic.Cbw:
                return TryWriteRegister(context, Register.AX, SignExtend(context[CpuRegister.Rax] & 0xFF, 8), 16, out failure);
            case Mnemonic.Cwde:
                return TryWriteRegister(context, Register.EAX, SignExtend(context[CpuRegister.Rax] & 0xFFFF, 16), 32, out failure);
            case Mnemonic.Cdqe:
                return TryWriteRegister(context, Register.RAX, SignExtend(context[CpuRegister.Rax] & 0xFFFF_FFFF, 32), 64, out failure);
            case Mnemonic.Cwd:
                return TryWriteRegister(context, Register.DX, (context[CpuRegister.Rax] & 0x8000) != 0 ? 0xFFFFUL : 0UL, 16, out failure);
            case Mnemonic.Cdq:
                return TryWriteRegister(context, Register.EDX, (context[CpuRegister.Rax] & 0x8000_0000) != 0 ? 0xFFFF_FFFFUL : 0UL, 32, out failure);
            case Mnemonic.Cqo:
                return TryWriteRegister(context, Register.RDX, (context[CpuRegister.Rax] & 0x8000_0000_0000_0000UL) != 0 ? ulong.MaxValue : 0UL, 64, out failure);
            default:
                failure = InterpreterFailure.Unsupported($"unsupported sign-extend mnemonic {mnemonic}");
                return false;
        }
    }

    private static bool ExecuteUnsignedDivide(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (bits is not (8 or 16 or 32 or 64))
        {
            failure = InterpreterFailure.Unsupported($"unsupported DIV size {bits}: {instruction}");
            return false;
        }

        if (!TryReadOperand(context, instruction, 0, bits, out var divisor, out failure))
        {
            return false;
        }
        if (divisor == 0)
        {
            failure = InterpreterFailure.Trap();
            return false;
        }

        var quotientLimit = (UInt128)X64Flags.Mask(bits);
        UInt128 dividend = bits switch
        {
            8 => context[CpuRegister.Rax] & 0xFFFF,
            16 => ((UInt128)(context[CpuRegister.Rdx] & 0xFFFF) << 16) | (context[CpuRegister.Rax] & 0xFFFF),
            32 => ((UInt128)(context[CpuRegister.Rdx] & 0xFFFF_FFFF) << 32) | (context[CpuRegister.Rax] & 0xFFFF_FFFF),
            64 => ((UInt128)context[CpuRegister.Rdx] << 64) | context[CpuRegister.Rax],
            _ => 0,
        };

        var quotient = dividend / divisor;
        var remainder = dividend % divisor;
        if (quotient > quotientLimit)
        {
            failure = InterpreterFailure.Trap();
            return false;
        }

        return bits switch
        {
            8 => TryWriteRegister(context, Register.AL, (ulong)quotient, 8, out failure) &&
                 TryWriteRegister(context, Register.AH, (ulong)remainder, 8, out failure),
            16 => TryWriteRegister(context, Register.AX, (ulong)quotient, 16, out failure) &&
                  TryWriteRegister(context, Register.DX, (ulong)remainder, 16, out failure),
            32 => TryWriteRegister(context, Register.EAX, (ulong)quotient, 32, out failure) &&
                  TryWriteRegister(context, Register.EDX, (ulong)remainder, 32, out failure),
            64 => TryWriteRegister(context, Register.RAX, (ulong)quotient, 64, out failure) &&
                  TryWriteRegister(context, Register.RDX, (ulong)remainder, 64, out failure),
            _ => UnsupportedDivisionSize(bits, out failure),
        };
    }

    private static bool ExecuteUnsignedMultiply(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (bits is not (8 or 16 or 32 or 64))
        {
            failure = InterpreterFailure.Unsupported($"unsupported MUL size {bits}: {instruction}");
            return false;
        }

        if (!TryReadOperand(context, instruction, 0, bits, out var multiplier, out failure))
        {
            return false;
        }

        UInt128 accumulator = context[CpuRegister.Rax] & X64Flags.Mask(bits);
        var product = accumulator * multiplier;
        var overflow = (product >> bits) != 0;

        context.Rflags = X64Flags.UpdateMultiplyOverflow(context.Rflags, overflow);

        return bits switch
        {
            8 => TryWriteRegister(context, Register.AX, (ulong)product, 16, out failure),
            16 => TryWriteRegister(context, Register.AX, (ulong)(product & 0xFFFF), 16, out failure) &&
                  TryWriteRegister(context, Register.DX, (ulong)(product >> 16), 16, out failure),
            32 => TryWriteRegister(context, Register.EAX, (ulong)(product & 0xFFFF_FFFF), 32, out failure) &&
                  TryWriteRegister(context, Register.EDX, (ulong)(product >> 32), 32, out failure),
            64 => TryWriteRegister(context, Register.RAX, (ulong)product, 64, out failure) &&
                  TryWriteRegister(context, Register.RDX, (ulong)(product >> 64), 64, out failure),
            _ => UnsupportedMultiplicationSize(bits, out failure),
        };
    }

    private static bool UnsupportedMultiplicationSize(int bits, out InterpreterFailure failure)
    {
        failure = InterpreterFailure.Unsupported($"unsupported multiplication size {bits}");
        return false;
    }

    private static bool ExecuteSignedMultiply(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.OpCount is not (2 or 3))
        {
            failure = InterpreterFailure.Unsupported($"unsupported IMUL operand count {instruction.OpCount}: {instruction}");
            return false;
        }

        var bits = GetOperandBitSize(instruction, 0);
        if (bits is not (16 or 32 or 64))
        {
            failure = InterpreterFailure.Unsupported($"unsupported IMUL size {bits}: {instruction}");
            return false;
        }

        var leftOperand = instruction.OpCount == 2 ? 0 : 1;
        var rightOperand = instruction.OpCount == 2 ? 1 : 2;
        if (!TryReadOperand(context, instruction, leftOperand, bits, out var leftRaw, out failure) ||
            !TryReadOperand(context, instruction, rightOperand, bits, out var rightRaw, out failure))
        {
            return false;
        }

        var product = ToSignedBigInteger(leftRaw, bits) * ToSignedBigInteger(rightRaw, bits);
        var mask = (BigInteger.One << bits) - BigInteger.One;
        var resultBig = product & mask;
        var result = (ulong)resultBig;
        var overflow = product != ToSignedBigInteger(result, bits);

        context.Rflags = X64Flags.UpdateMultiplyOverflow(context.Rflags, overflow);
        return TryWriteOperand(context, instruction, 0, result, bits, out failure);
    }

    private static BigInteger ToSignedBigInteger(ulong value, int bits)
    {
        value &= X64Flags.Mask(bits);
        var sign = 1UL << (bits - 1);
        if ((value & sign) == 0)
        {
            return value;
        }

        return (BigInteger)value - (BigInteger.One << bits);
    }

    private static BigInteger ToSignedBigInteger(BigInteger value, int bits)
    {
        var mask = (BigInteger.One << bits) - BigInteger.One;
        value &= mask;
        var sign = BigInteger.One << (bits - 1);
        return (value & sign) != 0 ? value - (BigInteger.One << bits) : value;
    }

    private static bool ExecuteSignedDivide(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (bits is not (8 or 16 or 32 or 64))
        {
            failure = InterpreterFailure.Unsupported($"unsupported IDIV size {bits}: {instruction}");
            return false;
        }

        if (!TryReadOperand(context, instruction, 0, bits, out var divisorRaw, out failure))
        {
            return false;
        }

        var divisor = ToSignedBigInteger(divisorRaw, bits);
        if (divisor.IsZero)
        {
            failure = InterpreterFailure.Trap();
            return false;
        }

        BigInteger dividendUnsigned = bits switch
        {
            8 => context[CpuRegister.Rax] & 0xFFFF,
            16 => ((BigInteger)(context[CpuRegister.Rdx] & 0xFFFF) << 16) | (context[CpuRegister.Rax] & 0xFFFF),
            32 => ((BigInteger)(context[CpuRegister.Rdx] & 0xFFFF_FFFF) << 32) | (context[CpuRegister.Rax] & 0xFFFF_FFFF),
            64 => ((BigInteger)context[CpuRegister.Rdx] << 64) | context[CpuRegister.Rax],
            _ => 0,
        };
        var dividendBits = bits == 8 ? 16 : bits * 2;
        var dividend = ToSignedBigInteger(dividendUnsigned, dividendBits);

        var quotient = dividend / divisor;
        var remainder = dividend - (quotient * divisor);

        var quotientLimit = BigInteger.One << (bits - 1);
        if (quotient >= quotientLimit || quotient < -quotientLimit)
        {
            failure = InterpreterFailure.Trap();
            return false;
        }

        var mask = (BigInteger)X64Flags.Mask(bits);
        var quotientBits = (ulong)(quotient & mask);
        var remainderBits = (ulong)(remainder & mask);

        return bits switch
        {
            8 => TryWriteRegister(context, Register.AL, quotientBits, 8, out failure) &&
                 TryWriteRegister(context, Register.AH, remainderBits, 8, out failure),
            16 => TryWriteRegister(context, Register.AX, quotientBits, 16, out failure) &&
                  TryWriteRegister(context, Register.DX, remainderBits, 16, out failure),
            32 => TryWriteRegister(context, Register.EAX, quotientBits, 32, out failure) &&
                  TryWriteRegister(context, Register.EDX, remainderBits, 32, out failure),
            64 => TryWriteRegister(context, Register.RAX, quotientBits, 64, out failure) &&
                  TryWriteRegister(context, Register.RDX, remainderBits, 64, out failure),
            _ => UnsupportedDivisionSize(bits, out failure),
        };
    }

    private static bool UnsupportedDivisionSize(int bits, out InterpreterFailure failure)
    {
        failure = InterpreterFailure.Unsupported($"unsupported division size {bits}");
        return false;
    }

    private static bool TryWriteAccumulator(CpuContext context, ulong value, int bits, out InterpreterFailure failure) =>
        bits switch
        {
            8 => TryWriteRegister(context, Register.AL, value, bits, out failure),
            16 => TryWriteRegister(context, Register.AX, value, bits, out failure),
            32 => TryWriteRegister(context, Register.EAX, value, bits, out failure),
            64 => TryWriteRegister(context, Register.RAX, value, bits, out failure),
            _ => UnsupportedAccumulatorSize(bits, out failure),
        };

    private static bool UnsupportedAccumulatorSize(int bits, out InterpreterFailure failure)
    {
        failure = InterpreterFailure.Unsupported($"unsupported accumulator size {bits}");
        return false;
    }

    private static bool ExecuteVectorMove(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var destinationKind = instruction.GetOpKind(0);
        var sourceKind = instruction.GetOpKind(1);
        if (destinationKind == OpKind.Register &&
            sourceKind == OpKind.Memory &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var destinationSize))
        {
            if (!TryGetMemoryAddress(context, instruction, out var address))
            {
                failure = InterpreterFailure.Unsupported("unsupported VMOVUPS memory address form");
                return false;
            }

            Span<byte> bytes = stackalloc byte[32];
            if (!TryReadInterpreterMemory(context, address, bytes[..destinationSize]))
            {
                failure = InterpreterFailure.MemoryRead(address, destinationSize);
                return false;
            }

            SetVectorRegister(context, destinationIndex, bytes[..destinationSize]);
            failure = default;
            return true;
        }

        if (destinationKind == OpKind.Memory &&
            sourceKind == OpKind.Register &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out var sourceSize))
        {
            if (!TryGetMemoryAddress(context, instruction, out var address))
            {
                failure = InterpreterFailure.Unsupported("unsupported VMOVUPS memory address form");
                return false;
            }

            Span<byte> bytes = stackalloc byte[32];
            GetVectorRegister(context, sourceIndex, sourceSize, bytes[..sourceSize]);
            if (!TryWriteInterpreterMemory(context, address, bytes[..sourceSize]))
            {
                failure = InterpreterFailure.MemoryWrite(address, sourceSize);
                return false;
            }

            failure = default;
            return true;
        }

        if (destinationKind == OpKind.Register &&
            sourceKind == OpKind.Register &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var copyDestinationIndex, out var copySize) &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var copySourceIndex, out _))
        {
            Span<byte> bytes = stackalloc byte[32];
            GetVectorRegister(context, copySourceIndex, copySize, bytes[..copySize]);
            SetVectorRegister(context, copyDestinationIndex, bytes[..copySize]);
            failure = default;
            return true;
        }

        failure = InterpreterFailure.Unsupported($"unsupported VMOVUPS operands: {instruction}");
        return false;
    }

    private static bool ExecuteVectorMoveHighPackedSingle(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.OpCount == 3 &&
            instruction.GetOpKind(0) == OpKind.Register &&
            instruction.GetOpKind(1) == OpKind.Register &&
            instruction.GetOpKind(2) == OpKind.Memory &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _) &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
        {
            if (!TryGetMemoryAddress(context, instruction, out var address))
            {
                failure = InterpreterFailure.Unsupported("unsupported VMOVHPS memory address form");
                return false;
            }

            Span<byte> bytes = stackalloc byte[16];
            GetVectorRegister(context, sourceIndex, 16, bytes);
            if (!TryReadInterpreterMemory(context, address, bytes[8..16]))
            {
                failure = InterpreterFailure.MemoryRead(address, sizeof(ulong));
                return false;
            }

            SetVectorRegister(context, destinationIndex, bytes);
            failure = default;
            return true;
        }

        if (instruction.OpCount == 2 &&
            instruction.GetOpKind(0) == OpKind.Memory &&
            instruction.GetOpKind(1) == OpKind.Register &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var storeSourceIndex, out _))
        {
            if (!TryGetMemoryAddress(context, instruction, out var address))
            {
                failure = InterpreterFailure.Unsupported("unsupported VMOVHPS memory address form");
                return false;
            }

            Span<byte> bytes = stackalloc byte[16];
            GetVectorRegister(context, storeSourceIndex, 16, bytes);
            if (!TryWriteInterpreterMemory(context, address, bytes[8..16]))
            {
                failure = InterpreterFailure.MemoryWrite(address, sizeof(ulong));
                return false;
            }

            failure = default;
            return true;
        }

        failure = InterpreterFailure.Unsupported($"unsupported VMOVHPS operands: {instruction}");
        return false;
    }

    private static bool ExecuteVectorMoveD(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var destinationKind = instruction.GetOpKind(0);
        var sourceKind = instruction.GetOpKind(1);
        if (destinationKind == OpKind.Register &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _))
        {
            if (!TryReadOperand(context, instruction, 1, 32, out var value, out failure))
            {
                return false;
            }

            context.SetXmmRegister(destinationIndex, (uint)value, 0);
            context.ClearYmmUpper(destinationIndex);
            failure = default;
            return true;
        }

        if (sourceKind == OpKind.Register &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
        {
            context.GetXmmRegister(sourceIndex, out var value, out _);
            return TryWriteOperand(context, instruction, 0, (uint)value, 32, out failure);
        }

        failure = InterpreterFailure.Unsupported($"unsupported VMOVD operands: {instruction}");
        return false;
    }

    private static bool ExecuteMoveScalar(
        CpuContext context,
        in Instruction instruction,
        bool doublePrecision,
        out InterpreterFailure failure)
    {
        var elementBytes = doublePrecision ? 8 : 4;

        if (instruction.GetOpKind(0) == OpKind.Memory)
        {
            if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var storeSourceIndex, out _))
            {
                failure = InterpreterFailure.Unsupported($"unsupported VMOVSS/VMOVSD store source: {instruction}");
                return false;
            }

            if (!TryGetMemoryAddress(context, instruction, out var storeAddress))
            {
                failure = InterpreterFailure.Unsupported("unsupported VMOVSS/VMOVSD memory address form");
                return false;
            }

            Span<byte> storeBytes = stackalloc byte[16];
            GetVectorRegister(context, storeSourceIndex, 16, storeBytes);
            if (!TryWriteInterpreterMemory(context, storeAddress, storeBytes[..elementBytes]))
            {
                failure = InterpreterFailure.MemoryWrite(storeAddress, elementBytes);
                return false;
            }

            failure = default;
            return true;
        }

        if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VMOVSS/VMOVSD destination: {instruction}");
            return false;
        }

        if (instruction.OpCount == 2)
        {
            if (!TryGetMemoryAddress(context, instruction, out var loadAddress))
            {
                failure = InterpreterFailure.Unsupported("unsupported VMOVSS/VMOVSD memory address form");
                return false;
            }

            Span<byte> loaded = stackalloc byte[16];
            if (!TryReadInterpreterMemory(context, loadAddress, loaded[..elementBytes]))
            {
                failure = InterpreterFailure.MemoryRead(loadAddress, elementBytes);
                return false;
            }

            SetVectorRegister(context, destinationIndex, loaded);
            failure = default;
            return true;
        }

        if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var mergeIndex, out _) ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(2), out var valueIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VMOVSS/VMOVSD register operands: {instruction}");
            return false;
        }

        context.GetXmmRegister(mergeIndex, out var mergeLow, out var mergeHigh);
        context.GetXmmRegister(valueIndex, out var valueLow, out _);

        var newLow = doublePrecision ? valueLow : (mergeLow & 0xFFFFFFFF00000000UL) | (valueLow & 0xFFFFFFFFUL);
        context.SetXmmRegister(destinationIndex, newLow, mergeHigh);
        context.ClearYmmUpper(destinationIndex);
        failure = default;
        return true;
    }

    private static bool ExecuteConvertIntToScalar(
        CpuContext context,
        in Instruction instruction,
        bool doublePrecision,
        out InterpreterFailure failure)
    {
        // Legacy: cvtsi2ss xmm, r/m32 and cvtsi2sd xmm, r/m32/r/m64.
        // VEX:    vcvtsi2ss xmm, xmm, r/m32 and vcvtsi2sd xmm, xmm, r/m32/r/m64.
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported scalar int-to-float destination: {instruction}");
            return false;
        }

        var sourceIndex = instruction.OpCount == 2 ? 1 : 2;
        var sourceBits = GetOperandBitSize(instruction, sourceIndex);
        if (!TryReadOperand(context, instruction, sourceIndex, sourceBits, out var rawValue, out failure))
            return false;

        var signedValue = sourceBits == 64 ? (long)rawValue : (int)(uint)rawValue;
        ulong mergeLow = 0;
        ulong mergeHigh = 0;
        if (instruction.OpCount == 3)
        {
            if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var mergeIndex, out _))
            {
                failure = InterpreterFailure.Unsupported($"unsupported scalar int-to-float merge operand: {instruction}");
                return false;
            }
            context.GetXmmRegister(mergeIndex, out mergeLow, out mergeHigh);
        }
        else
        {
            context.GetXmmRegister(destinationIndex, out mergeLow, out mergeHigh);
        }

        ulong newLow = doublePrecision
            ? BitConverter.DoubleToUInt64Bits(signedValue)
            : (mergeLow & 0xFFFFFFFF00000000UL) | BitConverter.SingleToUInt32Bits((float)signedValue);

        context.SetXmmRegister(destinationIndex, newLow, mergeHigh);
        context.ClearYmmUpper(destinationIndex);
        failure = default;
        return true;
    }

    private static bool ExecuteConvertScalarToInt(
        CpuContext context,
        in Instruction instruction,
        bool doublePrecision,
        bool truncate,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register)
        {
            failure = InterpreterFailure.Unsupported($"unsupported scalar float-to-int destination: {instruction}");
            return false;
        }

        var destinationBits = GetOperandBitSize(instruction, 0);

        long result;
        if (doublePrecision)
        {
            if (!TryReadScalarDouble(context, instruction, 1, out var value, out failure))
            {
                return false;
            }

            var converted = truncate ? Math.Truncate(value) : Math.Round(value, MidpointRounding.ToEven);
            result = destinationBits == 64 ? (long)converted : (int)converted;
        }
        else
        {
            if (!TryReadScalarSingle(context, instruction, 1, out var value, out failure))
            {
                return false;
            }

            var converted = truncate ? MathF.Truncate(value) : MathF.Round(value, MidpointRounding.ToEven);
            result = destinationBits == 64 ? (long)converted : (int)converted;
        }

        return TryWriteOperand(context, instruction, 0, unchecked((ulong)result), destinationBits, out failure);
    }

    private static bool ExecuteScalarArith(
        CpuContext context,
        in Instruction instruction,
        ScalarArithOp op,
        bool doublePrecision,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _) ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var leftIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported scalar arithmetic operands: {instruction}");
            return false;
        }

        context.GetXmmRegister(leftIndex, out var leftLow, out var leftHigh);

        if (doublePrecision)
        {
            if (!TryReadScalarDouble(context, instruction, 2, out var rightValue, out failure))
            {
                return false;
            }

            var leftValue = BitConverter.UInt64BitsToDouble(leftLow);
            var result = op switch
            {
                ScalarArithOp.Add => leftValue + rightValue,
                ScalarArithOp.Sub => leftValue - rightValue,
                ScalarArithOp.Mul => leftValue * rightValue,
                ScalarArithOp.Div => leftValue / rightValue,
                ScalarArithOp.Min => EvaluateMin(leftValue, rightValue),
                ScalarArithOp.Max => EvaluateMax(leftValue, rightValue),
                _ => leftValue,
            };
            context.SetXmmRegister(destinationIndex, BitConverter.DoubleToUInt64Bits(result), leftHigh);
        }
        else
        {
            if (!TryReadScalarSingle(context, instruction, 2, out var rightValue, out failure))
            {
                return false;
            }

            var leftValue = BitConverter.UInt32BitsToSingle((uint)leftLow);
            var result = op switch
            {
                ScalarArithOp.Add => leftValue + rightValue,
                ScalarArithOp.Sub => leftValue - rightValue,
                ScalarArithOp.Mul => leftValue * rightValue,
                ScalarArithOp.Div => leftValue / rightValue,
                ScalarArithOp.Min => EvaluateMinSingle(leftValue, rightValue),
                ScalarArithOp.Max => EvaluateMaxSingle(leftValue, rightValue),
                _ => leftValue,
            };
            var newLow = (leftLow & 0xFFFFFFFF00000000UL) | BitConverter.SingleToUInt32Bits(result);
            context.SetXmmRegister(destinationIndex, newLow, leftHigh);
        }

        context.ClearYmmUpper(destinationIndex);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorArith(
        CpuContext context,
        in Instruction instruction,
        ScalarArithOp op,
        bool doublePrecision,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported packed float arithmetic operands: {instruction}");
            return false;
        }

        Span<byte> left = stackalloc byte[32];
        Span<byte> right = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, left[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, right[..size], out failure))
        {
            return false;
        }

        var elementSize = doublePrecision ? 8 : 4;
        Span<byte> result = stackalloc byte[32];
        for (var offset = 0; offset < size; offset += elementSize)
        {
            if (doublePrecision)
            {
                var leftValue = BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(left.Slice(offset)));
                var rightValue = BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(right.Slice(offset)));
                var value = op switch
                {
                    ScalarArithOp.Add => leftValue + rightValue,
                    ScalarArithOp.Sub => leftValue - rightValue,
                    ScalarArithOp.Mul => leftValue * rightValue,
                    ScalarArithOp.Div => leftValue / rightValue,
                    ScalarArithOp.Min => EvaluateMin(leftValue, rightValue),
                    ScalarArithOp.Max => EvaluateMax(leftValue, rightValue),
                    _ => leftValue,
                };
                BinaryPrimitives.WriteUInt64LittleEndian(result.Slice(offset), BitConverter.DoubleToUInt64Bits(value));
            }
            else
            {
                var leftValue = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(left.Slice(offset)));
                var rightValue = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(right.Slice(offset)));
                var value = op switch
                {
                    ScalarArithOp.Add => leftValue + rightValue,
                    ScalarArithOp.Sub => leftValue - rightValue,
                    ScalarArithOp.Mul => leftValue * rightValue,
                    ScalarArithOp.Div => leftValue / rightValue,
                    ScalarArithOp.Min => EvaluateMinSingle(leftValue, rightValue),
                    ScalarArithOp.Max => EvaluateMaxSingle(leftValue, rightValue),
                    _ => leftValue,
                };
                BinaryPrimitives.WriteUInt32LittleEndian(result.Slice(offset), BitConverter.SingleToUInt32Bits(value));
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteScalarCompare(
        CpuContext context,
        in Instruction instruction,
        bool doublePrecision,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var leftIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported scalar compare operands: {instruction}");
            return false;
        }

        context.GetXmmRegister(leftIndex, out var leftLow, out _);

        int comparison;
        bool unordered;
        if (doublePrecision)
        {
            if (!TryReadScalarDouble(context, instruction, 1, out var rightValue, out failure))
            {
                return false;
            }

            var leftValue = BitConverter.UInt64BitsToDouble(leftLow);
            unordered = double.IsNaN(leftValue) || double.IsNaN(rightValue);
            comparison = unordered ? 0 : leftValue.CompareTo(rightValue);
        }
        else
        {
            if (!TryReadScalarSingle(context, instruction, 1, out var rightValue, out failure))
            {
                return false;
            }

            var leftValue = BitConverter.UInt32BitsToSingle((uint)leftLow);
            unordered = float.IsNaN(leftValue) || float.IsNaN(rightValue);
            comparison = unordered ? 0 : leftValue.CompareTo(rightValue);
        }

        const ulong carry = 1UL << 0;
        const ulong parity = 1UL << 2;
        const ulong zero = 1UL << 6;
        const ulong sign = 1UL << 7;
        const ulong overflow = 1UL << 11;
        context.Rflags &= ~(carry | parity | zero | sign | overflow);

        if (unordered)
        {
            context.Rflags |= carry | parity | zero;
        }
        else if (comparison < 0)
        {
            context.Rflags |= carry;
        }
        else if (comparison == 0)
        {
            context.Rflags |= zero;
        }

        failure = default;
        return true;
    }

    private static bool ExecuteScalarCompareToMask(
        CpuContext context,
        in Instruction instruction,
        bool doublePrecision,
        out InterpreterFailure failure)
    {
        var opCount = instruction.OpCount;
        var leftOpIndex = opCount == 4 ? 1 : 0;
        var rightOpIndex = opCount == 4 ? 2 : 1;
        var immOpIndex = opCount - 1;

        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _) ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(leftOpIndex), out var leftIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported scalar compare-to-mask operands: {instruction}");
            return false;
        }

        context.GetXmmRegister(leftIndex, out var leftLow, out var leftHigh);
        var predicate = (int)instruction.GetImmediate(immOpIndex);

        ulong newLow;
        if (doublePrecision)
        {
            if (!TryReadScalarDouble(context, instruction, rightOpIndex, out var rightValue, out failure))
            {
                return false;
            }

            var leftValue = BitConverter.UInt64BitsToDouble(leftLow);
            var unordered = double.IsNaN(leftValue) || double.IsNaN(rightValue);
            var comparison = unordered ? 0 : leftValue.CompareTo(rightValue);
            newLow = EvaluateFloatComparePredicate(predicate, comparison, unordered) ? ulong.MaxValue : 0UL;
        }
        else
        {
            if (!TryReadScalarSingle(context, instruction, rightOpIndex, out var rightValue, out failure))
            {
                return false;
            }

            var leftValue = BitConverter.UInt32BitsToSingle((uint)leftLow);
            var unordered = float.IsNaN(leftValue) || float.IsNaN(rightValue);
            var comparison = unordered ? 0 : leftValue.CompareTo(rightValue);
            var mask = EvaluateFloatComparePredicate(predicate, comparison, unordered) ? uint.MaxValue : 0U;
            newLow = (leftLow & 0xFFFFFFFF00000000UL) | mask;
        }

        context.SetXmmRegister(destinationIndex, newLow, leftHigh);
        context.ClearYmmUpper(destinationIndex);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorCompareToMask(
        CpuContext context,
        in Instruction instruction,
        bool doublePrecision,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported packed compare-to-mask destination: {instruction}");
            return false;
        }

        var opCount = instruction.OpCount;
        var leftOpIndex = opCount == 4 ? 1 : 0;
        var rightOpIndex = opCount == 4 ? 2 : 1;
        var immOpIndex = opCount - 1;

        Span<byte> left = stackalloc byte[32];
        Span<byte> right = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, leftOpIndex, size, left[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, rightOpIndex, size, right[..size], out failure))
        {
            return false;
        }

        var predicate = (int)instruction.GetImmediate(immOpIndex);
        var elementSize = doublePrecision ? 8 : 4;
        Span<byte> result = stackalloc byte[32];
        for (var offset = 0; offset < size; offset += elementSize)
        {
            bool matched;
            if (doublePrecision)
            {
                var leftValue = BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(left.Slice(offset)));
                var rightValue = BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(right.Slice(offset)));
                var unordered = double.IsNaN(leftValue) || double.IsNaN(rightValue);
                var comparison = unordered ? 0 : leftValue.CompareTo(rightValue);
                matched = EvaluateFloatComparePredicate(predicate, comparison, unordered);
                BinaryPrimitives.WriteUInt64LittleEndian(result.Slice(offset), matched ? ulong.MaxValue : 0UL);
            }
            else
            {
                var leftValue = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(left.Slice(offset)));
                var rightValue = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(right.Slice(offset)));
                var unordered = float.IsNaN(leftValue) || float.IsNaN(rightValue);
                var comparison = unordered ? 0 : leftValue.CompareTo(rightValue);
                matched = EvaluateFloatComparePredicate(predicate, comparison, unordered);
                BinaryPrimitives.WriteUInt32LittleEndian(result.Slice(offset), matched ? uint.MaxValue : 0U);
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorBlendVariable(
        CpuContext context,
        in Instruction instruction,
        int elementBytes,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported variable blend destination: {instruction}");
            return false;
        }

        var opCount = instruction.OpCount;
        var leftOpIndex = opCount == 4 ? 1 : 0;
        var rightOpIndex = opCount == 4 ? 2 : 1;

        Span<byte> left = stackalloc byte[32];
        Span<byte> right = stackalloc byte[32];
        Span<byte> mask = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, leftOpIndex, size, left[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, rightOpIndex, size, right[..size], out failure))
        {
            return false;
        }

        if (opCount == 4)
        {
            if (!TryReadVectorOperand(context, instruction, 3, size, mask[..size], out failure))
            {
                return false;
            }
        }
        else
        {
            context.GetXmmRegister(0, out var maskLow, out var maskHigh);
            BinaryPrimitives.WriteUInt64LittleEndian(mask, maskLow);
            BinaryPrimitives.WriteUInt64LittleEndian(mask[8..], maskHigh);
        }

        Span<byte> result = stackalloc byte[32];
        for (var offset = 0; offset < size; offset += elementBytes)
        {
            var selectRight = (mask[offset + elementBytes - 1] & 0x80) != 0;
            (selectRight ? right : left).Slice(offset, elementBytes).CopyTo(result.Slice(offset, elementBytes));
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteConvertScalarPrecision(
        CpuContext context,
        in Instruction instruction,
        bool toDouble,
        out InterpreterFailure failure)
    {
        var opCount = instruction.OpCount;
        var mergeOpIndex = opCount == 3 ? 1 : 0;
        var sourceOpIndex = opCount == 3 ? 2 : 1;

        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _) ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(mergeOpIndex), out var mergeIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported scalar precision conversion operands: {instruction}");
            return false;
        }

        context.GetXmmRegister(mergeIndex, out var mergeLow, out var mergeHigh);

        ulong newLow;
        if (toDouble)
        {
            if (!TryReadScalarSingle(context, instruction, sourceOpIndex, out var value, out failure))
            {
                return false;
            }

            newLow = BitConverter.DoubleToUInt64Bits(value);
        }
        else
        {
            if (!TryReadScalarDouble(context, instruction, sourceOpIndex, out var value, out failure))
            {
                return false;
            }

            var floatBits = (ulong)BitConverter.SingleToUInt32Bits((float)value);
            newLow = (mergeLow & 0xFFFFFFFF00000000UL) | floatBits;
        }

        context.SetXmmRegister(destinationIndex, newLow, mergeHigh);
        context.ClearYmmUpper(destinationIndex);
        failure = default;
        return true;
    }

    // MINSS/MAXSS/MINSD/MAXSD/(V)MINPS/etc. return the second operand whenever
    // either input is NaN, and compare via IEEE '<'/'>' otherwise (which treats
    // +0/-0 as equal, so equal-valued operands also fall through to the second
    // operand) â€” this differs from Math.Min/Max, which special-case NaN and
    // signed zero differently. See Intel SDM Vol. 2, MINSS/MAXSS pseudocode.
    private static double EvaluateMin(double left, double right) =>
        double.IsNaN(left) || double.IsNaN(right) ? right : left < right ? left : right;

    private static double EvaluateMax(double left, double right) =>
        double.IsNaN(left) || double.IsNaN(right) ? right : left > right ? left : right;

    private static float EvaluateMinSingle(float left, float right) =>
        float.IsNaN(left) || float.IsNaN(right) ? right : left < right ? left : right;

    private static float EvaluateMaxSingle(float left, float right) =>
        float.IsNaN(left) || float.IsNaN(right) ? right : left > right ? left : right;

    private static double ApplyRoundingMode(double value, int roundControl) => roundControl switch
    {
        0 => Math.Round(value, MidpointRounding.ToEven),
        1 => Math.Floor(value),
        2 => Math.Ceiling(value),
        _ => Math.Truncate(value),
    };

    private static float ApplyRoundingModeSingle(float value, int roundControl) => roundControl switch
    {
        0 => MathF.Round(value, MidpointRounding.ToEven),
        1 => MathF.Floor(value),
        2 => MathF.Ceiling(value),
        _ => MathF.Truncate(value),
    };

    private static bool ExecuteRoundScalar(
        CpuContext context,
        in Instruction instruction,
        bool doublePrecision,
        out InterpreterFailure failure)
    {
        var opCount = instruction.OpCount;
        var mergeOpIndex = opCount == 4 ? 1 : 0;
        var sourceOpIndex = opCount == 4 ? 2 : 1;
        var immOpIndex = opCount - 1;

        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _) ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(mergeOpIndex), out var mergeIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported scalar round operands: {instruction}");
            return false;
        }

        context.GetXmmRegister(mergeIndex, out var mergeLow, out var mergeHigh);
        var roundControl = (int)instruction.GetImmediate(immOpIndex) & 0x3;

        ulong newLow;
        if (doublePrecision)
        {
            if (!TryReadScalarDouble(context, instruction, sourceOpIndex, out var value, out failure))
            {
                return false;
            }

            newLow = BitConverter.DoubleToUInt64Bits(ApplyRoundingMode(value, roundControl));
        }
        else
        {
            if (!TryReadScalarSingle(context, instruction, sourceOpIndex, out var value, out failure))
            {
                return false;
            }

            var floatBits = (ulong)BitConverter.SingleToUInt32Bits(ApplyRoundingModeSingle(value, roundControl));
            newLow = (mergeLow & 0xFFFFFFFF00000000UL) | floatBits;
        }

        context.SetXmmRegister(destinationIndex, newLow, mergeHigh);
        context.ClearYmmUpper(destinationIndex);
        failure = default;
        return true;
    }

    private static bool ExecuteRoundPacked(
        CpuContext context,
        in Instruction instruction,
        bool doublePrecision,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported packed round destination: {instruction}");
            return false;
        }

        Span<byte> source = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, source[..size], out failure))
        {
            return false;
        }

        var roundControl = (int)instruction.GetImmediate(2) & 0x3;
        var elementSize = doublePrecision ? 8 : 4;
        Span<byte> result = stackalloc byte[32];
        for (var offset = 0; offset < size; offset += elementSize)
        {
            if (doublePrecision)
            {
                var value = BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(source.Slice(offset)));
                BinaryPrimitives.WriteUInt64LittleEndian(result.Slice(offset), BitConverter.DoubleToUInt64Bits(ApplyRoundingMode(value, roundControl)));
            }
            else
            {
                var value = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(offset)));
                BinaryPrimitives.WriteUInt32LittleEndian(result.Slice(offset), BitConverter.SingleToUInt32Bits(ApplyRoundingModeSingle(value, roundControl)));
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool TryGetVectorOperandSize(in Instruction instruction, int opIndex, out int size)
    {
        switch (instruction.GetOpKind(opIndex))
        {
            case OpKind.Register:
                return TryGetVectorRegisterInfo(instruction.GetOpRegister(opIndex), out _, out size);
            case OpKind.Memory:
                size = instruction.MemorySize.GetSize();
                return size is 16 or 32;
            default:
                size = 0;
                return false;
        }
    }

    private static bool ExecuteConvertPackedDoubleToSingle(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VCVTPD2PS destination: {instruction}");
            return false;
        }

        if (!TryGetVectorOperandSize(instruction, 1, out var sourceSize))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VCVTPD2PS source: {instruction}");
            return false;
        }

        Span<byte> source = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, sourceSize, source[..sourceSize], out failure))
        {
            return false;
        }

        var elementCount = sourceSize / 8;
        Span<byte> result = stackalloc byte[16];
        for (var i = 0; i < elementCount; i++)
        {
            var value = BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(source.Slice(i * 8)));
            BinaryPrimitives.WriteUInt32LittleEndian(result.Slice(i * 4), BitConverter.SingleToUInt32Bits((float)value));
        }

        SetVectorRegister(context, destinationIndex, result[..16]);
        failure = default;
        return true;
    }

    private static bool ExecuteConvertPackedSingleToDouble(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var destinationSize))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VCVTPS2PD destination: {instruction}");
            return false;
        }

        var elementCount = destinationSize / 8;
        var sourceBytes = elementCount * 4;

        // Register operands are always read at full xmm width (16 bytes) since
        // GetVectorRegister does not support partial-width reads; only memory
        // operands are read at the instruction's true (possibly 8-byte) size,
        // to avoid reading past a m64 operand's mapped region.
        Span<byte> source = stackalloc byte[16];
        var readSize = instruction.GetOpKind(1) == OpKind.Memory ? sourceBytes : 16;
        if (!TryReadVectorOperand(context, instruction, 1, readSize, source[..readSize], out failure))
        {
            return false;
        }

        Span<byte> result = stackalloc byte[32];
        for (var i = 0; i < elementCount; i++)
        {
            var value = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(i * 4)));
            BinaryPrimitives.WriteUInt64LittleEndian(result.Slice(i * 8), BitConverter.DoubleToUInt64Bits(value));
        }

        SetVectorRegister(context, destinationIndex, result[..destinationSize]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorExtract128(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(1) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out var sourceSize) ||
            sourceSize != 32)
        {
            failure = InterpreterFailure.Unsupported($"unsupported VEXTRACTI/F128 source: {instruction}");
            return false;
        }

        Span<byte> source = stackalloc byte[32];
        GetVectorRegister(context, sourceIndex, 32, source);

        var selector = (byte)instruction.GetImmediate(2);
        var half = (selector & 1) != 0 ? source.Slice(16, 16) : source.Slice(0, 16);

        if (instruction.GetOpKind(0) == OpKind.Register &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _))
        {
            SetVectorRegister(context, destinationIndex, half);
            failure = default;
            return true;
        }

        if (instruction.GetOpKind(0) == OpKind.Memory)
        {
            if (!TryGetMemoryAddress(context, instruction, out var address))
            {
                failure = InterpreterFailure.Unsupported("unsupported VEXTRACTI/F128 memory address form");
                return false;
            }

            if (!TryWriteInterpreterMemory(context, address, half))
            {
                failure = InterpreterFailure.MemoryWrite(address, 16);
                return false;
            }

            failure = default;
            return true;
        }

        failure = InterpreterFailure.Unsupported($"unsupported VEXTRACTI/F128 destination: {instruction}");
        return false;
    }

    private static bool ExecuteVectorInsert128(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var destinationSize) ||
            destinationSize != 32 ||
            instruction.GetOpKind(1) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VINSERTI/F128 operands: {instruction}");
            return false;
        }

        Span<byte> result = stackalloc byte[32];
        GetVectorRegister(context, sourceIndex, 32, result);

        Span<byte> insert = stackalloc byte[16];
        if (!TryReadVectorOperand(context, instruction, 2, 16, insert, out failure))
        {
            return false;
        }

        var selector = (byte)instruction.GetImmediate(3);
        var offset = (selector & 1) != 0 ? 16 : 0;
        insert.CopyTo(result.Slice(offset, 16));

        SetVectorRegister(context, destinationIndex, result);
        failure = default;
        return true;
    }

    // Boolean outcome of VCMPxx/CMPxx predicates 0-31 depends only on predicate & 0xF:
    // bits 16-31 repeat the semantics of 0-15 with different FP-exception signaling,
    // which this interpreter does not model. See Intel SDM Vol. 2, CMPPD/CMPSD predicate table.
    private static bool EvaluateFloatComparePredicate(int predicate, int comparison, bool unordered)
    {
        return (predicate & 0xF) switch
        {
            0 => !unordered && comparison == 0,
            1 => !unordered && comparison < 0,
            2 => !unordered && comparison <= 0,
            3 => unordered,
            4 => unordered || comparison != 0,
            5 => unordered || comparison >= 0,
            6 => unordered || comparison > 0,
            7 => !unordered,
            8 => unordered || comparison == 0,
            9 => unordered || comparison < 0,
            10 => unordered || comparison <= 0,
            11 => false,
            12 => !unordered && comparison != 0,
            13 => !unordered && comparison >= 0,
            14 => !unordered && comparison > 0,
            _ => true,
        };
    }

    private static bool TryReadScalarSingle(
        CpuContext context,
        in Instruction instruction,
        int opIndex,
        out float value,
        out InterpreterFailure failure)
    {
        value = 0;
        switch (instruction.GetOpKind(opIndex))
        {
            case OpKind.Register:
                if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(opIndex), out var index, out _))
                {
                    failure = InterpreterFailure.Unsupported($"unsupported scalar single register {instruction.GetOpRegister(opIndex)}");
                    return false;
                }

                context.GetXmmRegister(index, out var low, out _);
                value = BitConverter.UInt32BitsToSingle((uint)low);
                failure = default;
                return true;
            case OpKind.Memory:
                if (!TryGetMemoryAddress(context, instruction, out var address))
                {
                    failure = InterpreterFailure.Unsupported("unsupported scalar single memory address form");
                    return false;
                }

                Span<byte> singleBytes = stackalloc byte[sizeof(float)];
                if (!TryReadInterpreterMemory(context, address, singleBytes))
                {
                    failure = InterpreterFailure.MemoryRead(address, sizeof(float));
                    return false;
                }

                value = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(singleBytes));
                failure = default;
                return true;
            default:
                failure = InterpreterFailure.Unsupported($"unsupported scalar single operand kind {instruction.GetOpKind(opIndex)}");
                return false;
        }
    }

    private static bool TryReadScalarDouble(
        CpuContext context,
        in Instruction instruction,
        int opIndex,
        out double value,
        out InterpreterFailure failure)
    {
        value = 0;
        switch (instruction.GetOpKind(opIndex))
        {
            case OpKind.Register:
                if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(opIndex), out var index, out _))
                {
                    failure = InterpreterFailure.Unsupported($"unsupported scalar double register {instruction.GetOpRegister(opIndex)}");
                    return false;
                }

                context.GetXmmRegister(index, out var low, out _);
                value = BitConverter.UInt64BitsToDouble(low);
                failure = default;
                return true;
            case OpKind.Memory:
                if (!TryGetMemoryAddress(context, instruction, out var address))
                {
                    failure = InterpreterFailure.Unsupported("unsupported scalar double memory address form");
                    return false;
                }

                Span<byte> doubleBytes = stackalloc byte[sizeof(double)];
                if (!TryReadInterpreterMemory(context, address, doubleBytes))
                {
                    failure = InterpreterFailure.MemoryRead(address, sizeof(double));
                    return false;
                }

                value = BitConverter.UInt64BitsToDouble(BinaryPrimitives.ReadUInt64LittleEndian(doubleBytes));
                failure = default;
                return true;
            default:
                failure = InterpreterFailure.Unsupported($"unsupported scalar double operand kind {instruction.GetOpKind(opIndex)}");
                return false;
        }
    }

    private static bool ExecuteVectorMoveQ(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var destinationKind = instruction.GetOpKind(0);
        var sourceKind = instruction.GetOpKind(1);
        if (destinationKind == OpKind.Register &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _))
        {
            if (!TryReadOperand(context, instruction, 1, 64, out var value, out failure))
            {
                return false;
            }

            context.SetXmmRegister(destinationIndex, value, 0);
            context.ClearYmmUpper(destinationIndex);
            failure = default;
            return true;
        }

        if (sourceKind == OpKind.Register &&
            TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
        {
            context.GetXmmRegister(sourceIndex, out var value, out _);
            return TryWriteOperand(context, instruction, 0, value, 64, out failure);
        }

        failure = InterpreterFailure.Unsupported($"unsupported VMOVQ operands: {instruction}");
        return false;
    }

    private static bool ExecuteVectorExtractQ(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPEXTRQ source: {instruction}");
            return false;
        }

        context.GetXmmRegister(sourceIndex, out var low, out var high);
        var laneIndex = instruction.GetImmediate(2) & 0x1;
        var value = laneIndex == 0 ? low : high;

        return TryWriteOperand(context, instruction, 0, value, 64, out failure);
    }

    private static bool ExecuteVectorMoveExtend(
        CpuContext context,
        in Instruction instruction,
        int sourceElementBits,
        int destElementBits,
        bool signed,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var destinationSize))
        {
            failure = InterpreterFailure.Unsupported($"unsupported vector move-extend destination: {instruction}");
            return false;
        }

        var destElementSize = destElementBits / 8;
        var sourceElementSize = sourceElementBits / 8;
        var count = destinationSize / destElementSize;
        var sourceSize = count * sourceElementSize;

        Span<byte> source = stackalloc byte[32];
        switch (instruction.GetOpKind(1))
        {
            case OpKind.Register:
                if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
                {
                    failure = InterpreterFailure.Unsupported($"unsupported vector move-extend source: {instruction}");
                    return false;
                }

                // Register sources are always a full 128-bit XMM read; only
                // the low sourceSize bytes are meaningful for this op.
                GetVectorRegister(context, sourceIndex, 16, source[..16]);
                break;
            case OpKind.Memory:
                if (!TryGetMemoryAddress(context, instruction, out var address))
                {
                    failure = InterpreterFailure.Unsupported("unsupported vector move-extend memory address form");
                    return false;
                }

                if (!TryReadInterpreterMemory(context, address, source[..sourceSize]))
                {
                    failure = InterpreterFailure.MemoryRead(address, sourceSize);
                    return false;
                }

                break;
            default:
                failure = InterpreterFailure.Unsupported($"unsupported vector move-extend source operand: {instruction}");
                return false;
        }

        Span<byte> destination = stackalloc byte[32];
        for (var i = 0; i < count; i++)
        {
            ulong value = sourceElementBits switch
            {
                8 => source[i],
                16 => BinaryPrimitives.ReadUInt16LittleEndian(source[(i * 2)..]),
                32 => BinaryPrimitives.ReadUInt32LittleEndian(source[(i * 4)..]),
                _ => 0,
            };

            if (signed)
            {
                value = SignExtend(value, sourceElementBits);
            }

            switch (destElementBits)
            {
                case 16:
                    BinaryPrimitives.WriteUInt16LittleEndian(destination[(i * 2)..], (ushort)value);
                    break;
                case 32:
                    BinaryPrimitives.WriteUInt32LittleEndian(destination[(i * 4)..], (uint)value);
                    break;
                case 64:
                    BinaryPrimitives.WriteUInt64LittleEndian(destination[(i * 8)..], value);
                    break;
            }
        }

        SetVectorRegister(context, destinationIndex, destination[..destinationSize]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorExtractD(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPEXTRD source: {instruction}");
            return false;
        }

        Span<byte> bytes = stackalloc byte[16];
        GetVectorRegister(context, sourceIndex, 16, bytes);
        var laneIndex = (int)(instruction.GetImmediate(2) & 0x3);
        var value = BinaryPrimitives.ReadUInt32LittleEndian(bytes[(laneIndex * 4)..]);

        return TryWriteOperand(context, instruction, 0, value, 32, out failure);
    }

    private static bool ExecuteVectorInsertD(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out _) ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPINSRD operands: {instruction}");
            return false;
        }

        if (!TryReadOperand(context, instruction, 2, 32, out var insertValue, out failure))
        {
            return false;
        }

        Span<byte> bytes = stackalloc byte[16];
        GetVectorRegister(context, sourceIndex, 16, bytes);
        var laneIndex = (int)(instruction.GetImmediate(3) & 0x3);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes[(laneIndex * 4)..], (uint)insertValue);
        SetVectorRegister(context, destinationIndex, bytes);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorBroadcastQ(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var destinationSize))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPBROADCASTQ destination: {instruction}");
            return false;
        }

        ulong value;
        switch (instruction.GetOpKind(1))
        {
            case OpKind.Register:
                if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
                {
                    failure = InterpreterFailure.Unsupported($"unsupported VPBROADCASTQ source: {instruction.GetOpRegister(1)}");
                    return false;
                }

                context.GetXmmRegister(sourceIndex, out value, out _);
                break;
            case OpKind.Memory:
                if (!TryGetMemoryAddress(context, instruction, out var address))
                {
                    failure = InterpreterFailure.Unsupported("unsupported VPBROADCASTQ memory address form");
                    return false;
                }

                if (!context.TryReadUInt64(address, out value))
                {
                    failure = InterpreterFailure.MemoryRead(address, sizeof(ulong));
                    return false;
                }

                break;
            default:
                failure = InterpreterFailure.Unsupported($"unsupported VPBROADCASTQ source operand: {instruction}");
                return false;
        }

        Span<byte> bytes = stackalloc byte[32];
        for (var offset = 0; offset < destinationSize; offset += sizeof(ulong))
        {
            BinaryPrimitives.WriteUInt64LittleEndian(bytes[offset..], value);
        }

        SetVectorRegister(context, destinationIndex, bytes[..destinationSize]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorBroadcastD(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var destinationSize))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPBROADCASTD destination: {instruction}");
            return false;
        }

        uint value;
        switch (instruction.GetOpKind(1))
        {
            case OpKind.Register:
                if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
                {
                    failure = InterpreterFailure.Unsupported($"unsupported VPBROADCASTD source: {instruction.GetOpRegister(1)}");
                    return false;
                }

                context.GetXmmRegister(sourceIndex, out var sourceValue, out _);
                value = (uint)sourceValue;
                break;
            case OpKind.Memory:
                if (!TryGetMemoryAddress(context, instruction, out var address))
                {
                    failure = InterpreterFailure.Unsupported("unsupported VPBROADCASTD memory address form");
                    return false;
                }

                if (!context.TryReadUInt32(address, out value))
                {
                    failure = InterpreterFailure.MemoryRead(address, sizeof(uint));
                    return false;
                }

                break;
            default:
                failure = InterpreterFailure.Unsupported($"unsupported VPBROADCASTD source operand: {instruction}");
                return false;
        }

        Span<byte> bytes = stackalloc byte[32];
        for (var offset = 0; offset < destinationSize; offset += sizeof(uint))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes[offset..], value);
        }

        SetVectorRegister(context, destinationIndex, bytes[..destinationSize]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorBroadcastF128(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var destinationSize) ||
            destinationSize != 32)
        {
            failure = InterpreterFailure.Unsupported($"unsupported VBROADCASTF128 destination: {instruction}");
            return false;
        }

        if (!TryGetMemoryAddress(context, instruction, out var address))
        {
            failure = InterpreterFailure.Unsupported("unsupported VBROADCASTF128 memory address form");
            return false;
        }

        Span<byte> source = stackalloc byte[16];
        if (!TryReadInterpreterMemory(context, address, source))
        {
            failure = InterpreterFailure.MemoryRead(address, 16);
            return false;
        }

        Span<byte> result = stackalloc byte[32];
        source.CopyTo(result[..16]);
        source.CopyTo(result[16..]);
        SetVectorRegister(context, destinationIndex, result);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorBroadcastScalarSingle(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var destinationSize))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VBROADCASTSS destination: {instruction}");
            return false;
        }

        uint value;
        switch (instruction.GetOpKind(1))
        {
            case OpKind.Register:
                if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(1), out var sourceIndex, out _))
                {
                    failure = InterpreterFailure.Unsupported($"unsupported VBROADCASTSS source: {instruction.GetOpRegister(1)}");
                    return false;
                }

                context.GetXmmRegister(sourceIndex, out var low, out _);
                value = (uint)low;
                break;
            case OpKind.Memory:
                if (!TryGetMemoryAddress(context, instruction, out var address))
                {
                    failure = InterpreterFailure.Unsupported("unsupported VBROADCASTSS memory address form");
                    return false;
                }

                if (!context.TryReadUInt32(address, out value))
                {
                    failure = InterpreterFailure.MemoryRead(address, sizeof(uint));
                    return false;
                }

                break;
            default:
                failure = InterpreterFailure.Unsupported($"unsupported VBROADCASTSS source operand: {instruction}");
                return false;
        }

        Span<byte> bytes = stackalloc byte[32];
        for (var offset = 0; offset < destinationSize; offset += sizeof(uint))
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes[offset..], value);
        }

        SetVectorRegister(context, destinationIndex, bytes[..destinationSize]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorXor(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VXORPS destination: {instruction}");
            return false;
        }

        Span<byte> lhs = stackalloc byte[32];
        Span<byte> rhs = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, lhs[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, rhs[..size], out failure))
        {
            return false;
        }

        for (var i = 0; i < size; i++)
        {
            lhs[i] ^= rhs[i];
        }

        SetVectorRegister(context, destinationIndex, lhs[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorUnpack(
        CpuContext context,
        in Instruction instruction,
        UnpackHalf half,
        int elementBits,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported unpack destination: {instruction}");
            return false;
        }

        Span<byte> left = stackalloc byte[32];
        Span<byte> right = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, left[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, right[..size], out failure))
        {
            return false;
        }

        var elementSize = elementBits / 8;
        var elementsPerLane = 16 / elementSize;
        var halfCount = elementsPerLane / 2;
        var baseIndex = half == UnpackHalf.High ? halfCount : 0;

        Span<byte> result = stackalloc byte[32];
        for (var lane = 0; lane < size; lane += 16)
        {
            for (var i = 0; i < halfCount; i++)
            {
                var srcOffset = lane + ((baseIndex + i) * elementSize);
                var dstOffsetLeft = lane + (2 * i * elementSize);
                var dstOffsetRight = dstOffsetLeft + elementSize;
                left.Slice(srcOffset, elementSize).CopyTo(result.Slice(dstOffsetLeft, elementSize));
                right.Slice(srcOffset, elementSize).CopyTo(result.Slice(dstOffsetRight, elementSize));
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorPackedArith(
        CpuContext context,
        in Instruction instruction,
        PackedArithOp op,
        int elementBits,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported packed arithmetic destination: {instruction}");
            return false;
        }

        Span<byte> lhs = stackalloc byte[32];
        Span<byte> rhs = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, lhs[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, rhs[..size], out failure))
        {
            return false;
        }

        Span<byte> result = stackalloc byte[32];
        var elementSize = elementBits / 8;
        for (var i = 0; i < size; i += elementSize)
        {
            switch (elementBits)
            {
                case 8:
                    result[i] = op == PackedArithOp.Add ? (byte)(lhs[i] + rhs[i]) : (byte)(lhs[i] - rhs[i]);
                    break;
                case 16:
                {
                    var left = BinaryPrimitives.ReadUInt16LittleEndian(lhs[i..]);
                    var right = BinaryPrimitives.ReadUInt16LittleEndian(rhs[i..]);
                    BinaryPrimitives.WriteUInt16LittleEndian(
                        result[i..],
                        op == PackedArithOp.Add ? (ushort)(left + right) : (ushort)(left - right));
                    break;
                }
                case 32:
                {
                    var left = BinaryPrimitives.ReadUInt32LittleEndian(lhs[i..]);
                    var right = BinaryPrimitives.ReadUInt32LittleEndian(rhs[i..]);
                    BinaryPrimitives.WriteUInt32LittleEndian(
                        result[i..],
                        op == PackedArithOp.Add ? left + right : left - right);
                    break;
                }
                default:
                {
                    var left = BinaryPrimitives.ReadUInt64LittleEndian(lhs[i..]);
                    var right = BinaryPrimitives.ReadUInt64LittleEndian(rhs[i..]);
                    BinaryPrimitives.WriteUInt64LittleEndian(
                        result[i..],
                        op == PackedArithOp.Add ? left + right : left - right);
                    break;
                }
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorBitwise(
        CpuContext context,
        in Instruction instruction,
        VectorBitwiseOp op,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported vector bitwise destination: {instruction}");
            return false;
        }

        Span<byte> lhs = stackalloc byte[32];
        Span<byte> rhs = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, lhs[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, rhs[..size], out failure))
        {
            return false;
        }

        for (var i = 0; i < size; i++)
        {
            lhs[i] = op switch
            {
                VectorBitwiseOp.And => (byte)(lhs[i] & rhs[i]),
                VectorBitwiseOp.Or => (byte)(lhs[i] | rhs[i]),
                VectorBitwiseOp.AndNot => (byte)(~lhs[i] & rhs[i]),
                VectorBitwiseOp.Xor => (byte)(lhs[i] ^ rhs[i]),
                _ => lhs[i],
            };
        }

        SetVectorRegister(context, destinationIndex, lhs[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorPermuteLanes(
        CpuContext context,
        in Instruction instruction,
        int elementBytes,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPERMIL destination: {instruction}");
            return false;
        }

        Span<byte> source = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, source[..size], out failure))
        {
            return false;
        }

        var elementsPerLane = 16 / elementBytes;
        var selectBits = elementsPerLane == 2 ? 1 : 2;
        var selectMask = (1 << selectBits) - 1;
        var elementCount = size / elementBytes;

        Span<byte> result = stackalloc byte[32];
        if (instruction.OpCount == 3 && instruction.GetOpKind(2) == OpKind.Immediate8)
        {
            var imm = (byte)instruction.GetImmediate(2);
            for (var element = 0; element < elementCount; element++)
            {
                var localIndex = element % elementsPerLane;
                var lane = (element / elementsPerLane) * elementsPerLane;
                var selector = (imm >> (localIndex * selectBits)) & selectMask;
                source.Slice((lane + selector) * elementBytes, elementBytes)
                    .CopyTo(result.Slice(element * elementBytes, elementBytes));
            }
        }
        else
        {
            Span<byte> control = stackalloc byte[32];
            if (!TryReadVectorOperand(context, instruction, 2, size, control[..size], out failure))
            {
                return false;
            }

            for (var element = 0; element < elementCount; element++)
            {
                var lane = (element / elementsPerLane) * elementsPerLane;
                var controlOffset = element * elementBytes;
                var selector = elementBytes == 8
                    ? (int)((BinaryPrimitives.ReadUInt64LittleEndian(control.Slice(controlOffset)) >> 1) & 1)
                    : (int)(BinaryPrimitives.ReadUInt32LittleEndian(control.Slice(controlOffset)) & 3);
                source.Slice((lane + selector) * elementBytes, elementBytes)
                    .CopyTo(result.Slice(element * elementBytes, elementBytes));
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorShuffleDwords(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPSHUFD destination: {instruction}");
            return false;
        }

        Span<byte> source = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, source[..size], out failure))
        {
            return false;
        }

        var control = (byte)instruction.GetImmediate(2);
        Span<byte> result = stackalloc byte[32];
        for (var lane = 0; lane < size; lane += 16)
        {
            for (var element = 0; element < 4; element++)
            {
                var selector = (control >> (element * 2)) & 0x3;
                var value = BinaryPrimitives.ReadUInt32LittleEndian(source[(lane + (selector * 4))..]);
                BinaryPrimitives.WriteUInt32LittleEndian(result[(lane + (element * 4))..], value);
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorBlendDwords(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPBLENDD destination: {instruction}");
            return false;
        }

        Span<byte> left = stackalloc byte[32];
        Span<byte> right = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, left[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, right[..size], out failure))
        {
            return false;
        }

        var control = (byte)instruction.GetImmediate(3);
        Span<byte> result = stackalloc byte[32];
        var dwordCount = size / 4;
        for (var i = 0; i < dwordCount; i++)
        {
            var offset = i * 4;
            var source = ((control >> i) & 1) != 0 ? right : left;
            BinaryPrimitives.WriteUInt32LittleEndian(result[offset..], BinaryPrimitives.ReadUInt32LittleEndian(source[offset..]));
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorBlendWords(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPBLENDW destination: {instruction}");
            return false;
        }

        Span<byte> left = stackalloc byte[32];
        Span<byte> right = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, left[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, right[..size], out failure))
        {
            return false;
        }

        var control = (byte)instruction.GetImmediate(3);
        Span<byte> result = stackalloc byte[32];
        for (var lane = 0; lane < size; lane += 16)
        {
            for (var word = 0; word < 8; word++)
            {
                var offset = lane + (word * 2);
                var source = ((control >> word) & 1) != 0 ? right : left;
                BinaryPrimitives.WriteUInt16LittleEndian(
                    result[offset..],
                    BinaryPrimitives.ReadUInt16LittleEndian(source[offset..]));
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorShuffleBytes(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPSHUFB destination: {instruction}");
            return false;
        }

        Span<byte> source = stackalloc byte[32];
        Span<byte> control = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, source[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, control[..size], out failure))
        {
            return false;
        }

        Span<byte> result = stackalloc byte[32];
        for (var lane = 0; lane < size; lane += 16)
        {
            for (var offset = 0; offset < 16; offset++)
            {
                var selector = control[lane + offset];
                result[lane + offset] = (selector & 0x80) != 0
                    ? (byte)0
                    : source[lane + (selector & 0x0F)];
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorCompareEqualBytes(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPCMPEQB destination: {instruction}");
            return false;
        }

        Span<byte> lhs = stackalloc byte[32];
        Span<byte> rhs = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, lhs[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, rhs[..size], out failure))
        {
            return false;
        }

        Span<byte> result = stackalloc byte[32];
        for (var i = 0; i < size; i++)
        {
            result[i] = lhs[i] == rhs[i] ? (byte)0xFF : (byte)0;
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorCompareEqualDwords(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPCMPEQD destination: {instruction}");
            return false;
        }

        Span<byte> lhs = stackalloc byte[32];
        Span<byte> rhs = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, lhs[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, rhs[..size], out failure))
        {
            return false;
        }

        Span<byte> result = stackalloc byte[32];
        for (var i = 0; i < size; i += sizeof(uint))
        {
            var left = BinaryPrimitives.ReadUInt32LittleEndian(lhs[i..]);
            var right = BinaryPrimitives.ReadUInt32LittleEndian(rhs[i..]);
            BinaryPrimitives.WriteUInt32LittleEndian(result[i..], left == right ? uint.MaxValue : 0);
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorVariableShift(
        CpuContext context,
        in Instruction instruction,
        VariableShiftOp op,
        int elementBits,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported variable shift destination: {instruction}");
            return false;
        }

        Span<byte> values = stackalloc byte[32];
        Span<byte> counts = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, values[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 2, size, counts[..size], out failure))
        {
            return false;
        }

        Span<byte> result = stackalloc byte[32];
        var elementSize = elementBits / 8;
        for (var i = 0; i < size; i += elementSize)
        {
            if (elementBits == 32)
            {
                var value = BinaryPrimitives.ReadUInt32LittleEndian(values[i..]);
                var count = BinaryPrimitives.ReadUInt32LittleEndian(counts[i..]);
                var shifted = op switch
                {
                    VariableShiftOp.Srlv => count >= 32 ? 0U : value >> (int)count,
                    VariableShiftOp.Sllv => count >= 32 ? 0U : value << (int)count,
                    VariableShiftOp.Srav => unchecked((uint)(count >= 32 ? (int)value >> 31 : (int)value >> (int)count)),
                    _ => value,
                };
                BinaryPrimitives.WriteUInt32LittleEndian(result[i..], shifted);
            }
            else
            {
                var value = BinaryPrimitives.ReadUInt64LittleEndian(values[i..]);
                var count = BinaryPrimitives.ReadUInt64LittleEndian(counts[i..]);
                var shifted = op switch
                {
                    VariableShiftOp.Srlv => count >= 64 ? 0UL : value >> (int)count,
                    VariableShiftOp.Sllv => count >= 64 ? 0UL : value << (int)count,
                    VariableShiftOp.Srav => unchecked((ulong)(count >= 64 ? (long)value >> 63 : (long)value >> (int)count)),
                    _ => value,
                };
                BinaryPrimitives.WriteUInt64LittleEndian(result[i..], shifted);
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorPackedShift(
        CpuContext context,
        in Instruction instruction,
        PackedShiftOp op,
        int elementBits,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out var destinationIndex, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported packed shift destination: {instruction}");
            return false;
        }

        Span<byte> values = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 1, size, values[..size], out failure))
        {
            return false;
        }

        ulong count;
        var countKind = instruction.GetOpKind(2);
        if (countKind is OpKind.Register or OpKind.Memory)
        {
            // The shift count is the low 64 bits of a 128-bit operand,
            // applied uniformly to every lane (not a per-lane count).
            Span<byte> countBytes = stackalloc byte[16];
            if (!TryReadVectorOperand(context, instruction, 2, 16, countBytes, out failure))
            {
                return false;
            }

            count = BinaryPrimitives.ReadUInt64LittleEndian(countBytes);
        }
        else
        {
            count = instruction.GetImmediate(2);
        }

        Span<byte> result = stackalloc byte[32];
        var elementSize = elementBits / 8;
        for (var i = 0; i < size; i += elementSize)
        {
            switch (elementBits)
            {
                case 64:
                {
                    var value = BinaryPrimitives.ReadUInt64LittleEndian(values[i..]);
                    var shifted = op switch
                    {
                        PackedShiftOp.Sll => count >= 64 ? 0UL : value << (int)count,
                        PackedShiftOp.Srl => count >= 64 ? 0UL : value >> (int)count,
                        PackedShiftOp.Sra => unchecked((ulong)(count >= 64 ? (long)value >> 63 : (long)value >> (int)count)),
                        _ => value,
                    };
                    BinaryPrimitives.WriteUInt64LittleEndian(result[i..], shifted);
                    break;
                }
                case 32:
                {
                    var value = BinaryPrimitives.ReadUInt32LittleEndian(values[i..]);
                    var shifted = op switch
                    {
                        PackedShiftOp.Sll => count >= 32 ? 0U : value << (int)count,
                        PackedShiftOp.Srl => count >= 32 ? 0U : value >> (int)count,
                        PackedShiftOp.Sra => unchecked((uint)(count >= 32 ? (int)value >> 31 : (int)value >> (int)count)),
                        _ => value,
                    };
                    BinaryPrimitives.WriteUInt32LittleEndian(result[i..], shifted);
                    break;
                }
                default:
                {
                    var value = BinaryPrimitives.ReadUInt16LittleEndian(values[i..]);
                    var shifted = op switch
                    {
                        PackedShiftOp.Sll => count >= 16 ? (ushort)0 : (ushort)(value << (int)count),
                        PackedShiftOp.Srl => count >= 16 ? (ushort)0 : (ushort)(value >> (int)count),
                        PackedShiftOp.Sra => unchecked((ushort)(count >= 16 ? (short)value >> 15 : (short)value >> (int)count)),
                        _ => value,
                    };
                    BinaryPrimitives.WriteUInt16LittleEndian(result[i..], shifted);
                    break;
                }
            }
        }

        SetVectorRegister(context, destinationIndex, result[..size]);
        failure = default;
        return true;
    }

    private static bool ExecuteVectorTest(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out _, out var size))
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPTEST lhs: {instruction}");
            return false;
        }

        Span<byte> lhs = stackalloc byte[32];
        Span<byte> rhs = stackalloc byte[32];
        if (!TryReadVectorOperand(context, instruction, 0, size, lhs[..size], out failure) ||
            !TryReadVectorOperand(context, instruction, 1, size, rhs[..size], out failure))
        {
            return false;
        }

        var andIsZero = true;
        var andNotIsZero = true;
        for (var i = 0; i < size; i++)
        {
            andIsZero &= (lhs[i] & rhs[i]) == 0;
            andNotIsZero &= ((~lhs[i]) & rhs[i]) == 0;
        }

        const ulong carry = 1UL << 0;
        const ulong parity = 1UL << 2;
        const ulong adjust = 1UL << 4;
        const ulong zero = 1UL << 6;
        const ulong sign = 1UL << 7;
        const ulong overflow = 1UL << 11;
        const ulong affected = carry | parity | adjust | zero | sign | overflow;

        context.Rflags &= ~affected;
        if (andNotIsZero)
        {
            context.Rflags |= carry;
        }
        if (andIsZero)
        {
            context.Rflags |= zero;
        }

        failure = default;
        return true;
    }

    private static bool ExecuteVectorCompareImplicitStringIndex(
        CpuContext context,
        in Instruction instruction,
        out InterpreterFailure failure)
    {
        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetVectorRegisterInfo(instruction.GetOpRegister(0), out _, out var size) ||
            size != 16)
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPCMPISTRI operands: {instruction}");
            return false;
        }

        var control = (byte)instruction.GetImmediate(2);
        if (control == 0x38)
        {
            Span<byte> text = stackalloc byte[16];
            if (!TryReadVectorOperand(context, instruction, 1, size, text, out failure))
            {
                return false;
            }

            var nullIndex = text.IndexOf((byte)0);
            var nullResultIndex = nullIndex < 0 ? 16 : nullIndex;
            var nullMatchMask = nullIndex < 0 ? 0 : 1 << nullIndex;

            context[CpuRegister.Rcx] = (uint)nullResultIndex;
            ApplyPcmpistriFlags(context, nullMatchMask, ImplicitStringLength(text), ImplicitStringLength(text));
            failure = default;
            return true;
        }

        if (control != 0)
        {
            failure = InterpreterFailure.Unsupported($"unsupported VPCMPISTRI control 0x{control:X2}: {instruction}");
            return false;
        }

        Span<byte> set = stackalloc byte[16];
        Span<byte> value = stackalloc byte[16];
        if (!TryReadVectorOperand(context, instruction, 0, size, set, out failure) ||
            !TryReadVectorOperand(context, instruction, 1, size, value, out failure))
        {
            return false;
        }

        var setLength = ImplicitStringLength(set);
        var valueLength = ImplicitStringLength(value);
        var matchMask = 0;
        for (var valueIndex = 0; valueIndex < valueLength; valueIndex++)
        {
            for (var setIndex = 0; setIndex < setLength; setIndex++)
            {
                if (value[valueIndex] == set[setIndex])
                {
                    matchMask |= 1 << valueIndex;
                    break;
                }
            }
        }

        var index = 16;
        for (var i = 0; i < 16; i++)
        {
            if ((matchMask & (1 << i)) != 0)
            {
                index = i;
                break;
            }
        }

        context[CpuRegister.Rcx] = (uint)index;

        ApplyPcmpistriFlags(context, matchMask, valueLength, setLength);

        failure = default;
        return true;
    }

    private static void ApplyPcmpistriFlags(CpuContext context, int matchMask, int valueLength, int setLength)
    {
        const ulong carry = 1UL << 0;
        const ulong parity = 1UL << 2;
        const ulong adjust = 1UL << 4;
        const ulong zero = 1UL << 6;
        const ulong sign = 1UL << 7;
        const ulong overflow = 1UL << 11;
        const ulong affected = carry | parity | adjust | zero | sign | overflow;

        context.Rflags &= ~affected;
        if (matchMask != 0)
        {
            context.Rflags |= carry;
        }
        if (valueLength < 16)
        {
            context.Rflags |= zero;
        }
        if (setLength < 16)
        {
            context.Rflags |= sign;
        }
        if ((matchMask & 1) != 0)
        {
            context.Rflags |= overflow;
        }
    }

    private static int ImplicitStringLength(ReadOnlySpan<byte> bytes)
    {
        var terminator = bytes.IndexOf((byte)0);
        return terminator < 0 ? bytes.Length : terminator;
    }

    private static bool ExecuteLea(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (!TryGetMemoryAddress(context, instruction, out var address))
        {
            failure = InterpreterFailure.Unsupported("unsupported LEA address form");
            return false;
        }
        return TryWriteOperand(context, instruction, 0, address, GetOperandBitSize(instruction, 0), out failure);
    }

    private static bool ExecutePush(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (!TryReadOperand(context, instruction, 0, 64, out var value, out failure))
        {
            return false;
        }
        if (!context.PushUInt64(value))
        {
            failure = InterpreterFailure.MemoryWrite(context[CpuRegister.Rsp], sizeof(ulong));
            return false;
        }
        return true;
    }

    private static bool ExecutePop(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        if (!context.PopUInt64(out var value))
        {
            failure = InterpreterFailure.MemoryRead(context[CpuRegister.Rsp], sizeof(ulong));
            return false;
        }
        return TryWriteOperand(context, instruction, 0, value, GetOperandBitSize(instruction, 0), out failure);
    }

    private static bool ExecuteUnary(
        CpuContext context,
        in Instruction instruction,
        UnaryOp op,
        out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 0, bits, out var value, out failure))
        {
            return false;
        }

        var mask = X64Flags.Mask(bits);
        var result = op switch
        {
            UnaryOp.Inc => (value + 1) & mask,
            UnaryOp.Dec => (value - 1) & mask,
            UnaryOp.Neg => (0UL - value) & mask,
            UnaryOp.Not => (~value) & mask,
            _ => value,
        };

        // NOT is a pure bitwise complement: unlike NEG, it leaves every flag
        // untouched (Intel SDM, "NOT" flags-affected list is empty).
        if (op != UnaryOp.Not)
        {
            var oldCarry = context.Rflags & 1UL;
            var updatedFlags = op switch
            {
                UnaryOp.Inc => X64Flags.UpdateAdd(context.Rflags, value, 1, result, bits),
                UnaryOp.Dec => X64Flags.UpdateSub(context.Rflags, value, 1, result, bits),
                UnaryOp.Neg => X64Flags.UpdateSub(context.Rflags, 0, value, result, bits),
                _ => context.Rflags,
            };
            context.Rflags = op is UnaryOp.Inc or UnaryOp.Dec ? (updatedFlags & ~1UL) | oldCarry : updatedFlags;
        }

        return TryWriteOperand(context, instruction, 0, result, bits, out failure);
    }

    private static bool ExecuteAdc(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 0, bits, out var lhs, out failure) ||
            !TryReadOperand(context, instruction, 1, bits, out var rhs, out failure))
        {
            return false;
        }

        var mask = X64Flags.Mask(bits);
        var carryIn = context.Rflags & 1UL;
        var sum = (UInt128)(lhs & mask) + (rhs & mask) + carryIn;
        var result = (ulong)(sum & mask);

        // UpdateAdd's own carry check (result < lhs) only ever misses a
        // carry that the extra carry-in caused (never reports one that
        // isn't there), so OR-ing in the true wide-sum overflow is
        // sufficient to correct it without needing a separate flag path.
        context.Rflags = X64Flags.UpdateAdd(context.Rflags, lhs, rhs, result, bits);
        if (sum > mask)
        {
            context.Rflags |= 1UL;
        }

        return TryWriteOperand(context, instruction, 0, result, bits, out failure);
    }

    private static bool ExecuteSbb(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 0, bits, out var lhs, out failure) ||
            !TryReadOperand(context, instruction, 1, bits, out var rhs, out failure))
        {
            return false;
        }

        var mask = X64Flags.Mask(bits);
        var carryIn = context.Rflags & 1UL;
        var maskedLhs = (UInt128)(lhs & mask);
        var subtrahend = (UInt128)(rhs & mask) + carryIn;
        var borrow = maskedLhs < subtrahend;
        var result = unchecked((ulong)((maskedLhs - subtrahend) & mask));

        // Same reasoning as ExecuteAdc, mirrored: UpdateSub's (lhs < rhs)
        // check can only under-report the borrow the extra borrow-in
        // causes, never over-report it.
        context.Rflags = X64Flags.UpdateSub(context.Rflags, lhs, rhs, result, bits);
        if (borrow)
        {
            context.Rflags |= 1UL;
        }

        return TryWriteOperand(context, instruction, 0, result, bits, out failure);
    }

    private static bool ExecuteBinary(
        CpuContext context,
        in Instruction instruction,
        BinaryOp op,
        bool writeResult,
        out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 0, bits, out var lhs, out failure) ||
            !TryReadOperand(context, instruction, 1, bits, out var rhs, out failure))
        {
            return false;
        }

        var mask = X64Flags.Mask(bits);
        var result = op switch
        {
            BinaryOp.Add => (lhs + rhs) & mask,
            BinaryOp.Sub => (lhs - rhs) & mask,
            BinaryOp.Xor => (lhs ^ rhs) & mask,
            BinaryOp.And => (lhs & rhs) & mask,
            BinaryOp.Or => (lhs | rhs) & mask,
            _ => lhs,
        };

        context.Rflags = op switch
        {
            BinaryOp.Add => X64Flags.UpdateAdd(context.Rflags, lhs, rhs, result, bits),
            BinaryOp.Sub => X64Flags.UpdateSub(context.Rflags, lhs, rhs, result, bits),
            _ => X64Flags.UpdateLogic(context.Rflags, result, bits),
        };

        return !writeResult || TryWriteOperand(context, instruction, 0, result, bits, out failure);
    }

    private static bool ExecuteShift(
        CpuContext context,
        in Instruction instruction,
        ShiftOp op,
        out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 0, bits, out var value, out failure) ||
            !TryReadOperand(context, instruction, 1, GetOperandBitSize(instruction, 1), out var rawCount, out failure))
        {
            return false;
        }

        var count = (int)(rawCount & (bits == 64 ? 0x3FUL : 0x1FUL));
        if (count == 0)
        {
            return true;
        }

        var mask = X64Flags.Mask(bits);
        value &= mask;
        var result = op switch
        {
            ShiftOp.Shl => (value << count) & mask,
            ShiftOp.Shr => value >> count,
            ShiftOp.Sar => ArithmeticShiftRight(value, count, bits),
            _ => value,
        };

        var carry = op switch
        {
            ShiftOp.Shl when count <= bits => (value >> (bits - count)) & 1UL,
            ShiftOp.Shr or ShiftOp.Sar when count <= bits => (value >> (count - 1)) & 1UL,
            _ => 0UL,
        };
        var overflow = count == 1 && op switch
        {
            ShiftOp.Shl => (((result >> (bits - 1)) ^ carry) & 1UL) != 0,
            ShiftOp.Shr => (value & (1UL << (bits - 1))) != 0,
            ShiftOp.Sar => false,
            _ => false,
        };

        context.Rflags = X64Flags.UpdateLogic(context.Rflags, result, bits);
        if (carry != 0)
        {
            context.Rflags |= 1UL;
        }
        if (overflow)
        {
            context.Rflags |= 1UL << 11;
        }

        return TryWriteOperand(context, instruction, 0, result, bits, out failure);
    }

    private static bool ExecuteBitTest(
        CpuContext context,
        in Instruction instruction,
        out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 0, bits, out var value, out failure) ||
            !TryReadOperand(context, instruction, 1, GetOperandBitSize(instruction, 1), out var rawBit, out failure))
        {
            return false;
        }

        var bit = (int)(rawBit & (ulong)(bits - 1));
        var carry = (value >> bit) & 1UL;
        context.Rflags = (context.Rflags & ~1UL) | carry;
        failure = default;
        return true;
    }

    private static bool ExecuteShiftx(
        CpuContext context,
        in Instruction instruction,
        ShiftxOp op,
        out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 1, bits, out var value, out failure) ||
            !TryReadOperand(context, instruction, 2, bits, out var countValue, out failure))
        {
            return false;
        }

        var count = (int)(countValue & (ulong)(bits - 1));
        var mask = X64Flags.Mask(bits);
        var result = op switch
        {
            ShiftxOp.Shlx => (value << count) & mask,
            ShiftxOp.Shrx => (value & mask) >> count,
            ShiftxOp.Sarx => ArithmeticShiftRight(value, count, bits),
            _ => value & mask,
        };

        return TryWriteOperand(context, instruction, 0, result, bits, out failure);
    }

    private static bool ExecuteBitFieldExtract(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 1, bits, out var value, out failure) ||
            !TryReadOperand(context, instruction, 2, bits, out var control, out failure))
        {
            return false;
        }

        var start = (int)(control & 0xFF);
        var length = (int)((control >> 8) & 0xFF);
        var result = 0UL;
        if (start < bits && length > 0)
        {
            var effectiveLength = Math.Min(length, bits - start);
            var extractMask = effectiveLength >= 64 ? ulong.MaxValue : (1UL << effectiveLength) - 1;
            result = (value >> start) & extractMask;
        }

        // BEXTR clears CF/OF and sets ZF from the result; SF/AF/PF are
        // documented as undefined, so this mirrors what real hardware
        // guarantees rather than leaving stale flag bits in place.
        const ulong carry = 1UL << 0;
        const ulong parity = 1UL << 2;
        const ulong adjust = 1UL << 4;
        const ulong zero = 1UL << 6;
        const ulong sign = 1UL << 7;
        const ulong overflow = 1UL << 11;
        context.Rflags &= ~(carry | parity | adjust | zero | sign | overflow);
        if (result == 0)
        {
            context.Rflags |= zero;
        }

        return TryWriteOperand(context, instruction, 0, result, bits, out failure);
    }

    private static bool ExecuteByteSwap(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 0, bits, out var value, out failure))
        {
            return false;
        }

        var swapped = bits switch
        {
            32 => (ulong)BinaryPrimitives.ReverseEndianness((uint)value),
            64 => BinaryPrimitives.ReverseEndianness(value),
            _ => value,
        };

        return TryWriteOperand(context, instruction, 0, swapped, bits, out failure);
    }

    private static bool ExecuteZeroHighBits(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 1, bits, out var value, out failure) ||
            !TryReadOperand(context, instruction, 2, bits, out var indexOperand, out failure))
        {
            return false;
        }

        var index = (int)(indexOperand & 0xFF);
        var mask = X64Flags.Mask(bits);
        var outOfRange = index >= bits;
        var result = outOfRange
            ? value & mask
            : value & (index == 0 ? 0UL : (1UL << index) - 1) & mask;

        const ulong carry = 1UL << 0;
        const ulong parity = 1UL << 2;
        const ulong adjust = 1UL << 4;
        const ulong zero = 1UL << 6;
        const ulong sign = 1UL << 7;
        const ulong overflow = 1UL << 11;
        context.Rflags &= ~(carry | parity | adjust | zero | sign | overflow);
        if (outOfRange)
        {
            context.Rflags |= carry;
        }

        if (result == 0)
        {
            context.Rflags |= zero;
        }

        return TryWriteOperand(context, instruction, 0, result, bits, out failure);
    }

    private static bool ExecuteRotateRightNoFlags(CpuContext context, in Instruction instruction, out InterpreterFailure failure)
    {
        var bits = GetOperandBitSize(instruction, 0);
        if (!TryReadOperand(context, instruction, 1, bits, out var value, out failure))
        {
            return false;
        }

        var count = (int)(instruction.GetImmediate(2) & (ulong)(bits - 1));
        var mask = X64Flags.Mask(bits);
        value &= mask;
        var result = count == 0 ? value : ((value >> count) | (value << (bits - count))) & mask;
        return TryWriteOperand(context, instruction, 0, result, bits, out failure);
    }

    private static ulong ArithmeticShiftRight(ulong value, int count, int bits)
    {
        if (bits == 64)
        {
            return (ulong)((long)value >> count);
        }

        var shift = 64 - bits;
        return (ulong)(((long)(value << shift) >> shift) >> count) & X64Flags.Mask(bits);
    }

    private static bool TryGetBranchTarget(
        CpuContext context,
        in Instruction instruction,
        out ulong target,
        out InterpreterFailure failure)
    {
        failure = default;
        target = 0;
        var opKind = instruction.GetOpKind(0);
        if (opKind is OpKind.NearBranch16 or OpKind.NearBranch32 or OpKind.NearBranch64)
        {
            target = instruction.NearBranch64;
            return true;
        }
        return TryReadOperand(context, instruction, 0, 64, out target, out failure);
    }

    private static bool TryReadOperand(
        CpuContext context,
        in Instruction instruction,
        int opIndex,
        int bits,
        out ulong value,
        out InterpreterFailure failure)
    {
        failure = default;
        value = 0;
        switch (instruction.GetOpKind(opIndex))
        {
            case OpKind.Register:
                return TryReadRegister(context, instruction.GetOpRegister(opIndex), bits, out value, out failure);
            case OpKind.Immediate8:
            case OpKind.Immediate8to16:
            case OpKind.Immediate8to32:
            case OpKind.Immediate8to64:
            case OpKind.Immediate16:
            case OpKind.Immediate32:
            case OpKind.Immediate32to64:
            case OpKind.Immediate64:
                value = instruction.GetImmediate(opIndex) & X64Flags.Mask(bits);
                return true;
            case OpKind.Memory:
                if (!TryGetMemoryAddress(context, instruction, out var address))
                {
                    failure = InterpreterFailure.Unsupported("unsupported memory address form");
                    return false;
                }
                return TryReadMemory(context, address, bits, out value, out failure);
            default:
                failure = InterpreterFailure.Unsupported($"unsupported operand kind {instruction.GetOpKind(opIndex)}");
                return false;
        }
    }

    private static bool TryWriteOperand(
        CpuContext context,
        in Instruction instruction,
        int opIndex,
        ulong value,
        int bits,
        out InterpreterFailure failure)
    {
        failure = default;
        value &= X64Flags.Mask(bits);
        switch (instruction.GetOpKind(opIndex))
        {
            case OpKind.Register:
                return TryWriteRegister(context, instruction.GetOpRegister(opIndex), value, bits, out failure);
            case OpKind.Memory:
                if (!TryGetMemoryAddress(context, instruction, out var address))
                {
                    failure = InterpreterFailure.Unsupported("unsupported memory address form");
                    return false;
                }
                return TryWriteMemory(context, address, value, bits, out failure);
            default:
                failure = InterpreterFailure.Unsupported($"unsupported destination operand {instruction.GetOpKind(opIndex)}");
                return false;
        }
    }

    private static bool TryGetMemoryAddress(CpuContext context, in Instruction instruction, out ulong address)
    {
        address = instruction.MemoryDisplacement64;
        if (instruction.IsIPRelativeMemoryOperand)
        {
            address = instruction.IPRelativeMemoryAddress;
            return true;
        }
        if (TryGetBaseRegisterValue(context, instruction.MemoryBase, out var baseValue))
        {
            address += baseValue;
        }
        else if (instruction.MemoryBase != Register.None)
        {
            return false;
        }
        if (TryGetBaseRegisterValue(context, instruction.MemoryIndex, out var indexValue))
        {
            address += indexValue * (ulong)instruction.MemoryIndexScale;
        }
        else if (instruction.MemoryIndex != Register.None)
        {
            return false;
        }
        if (instruction.MemorySegment == Register.FS)
        {
            address += context.FsBase;
        }
        else if (instruction.MemorySegment == Register.GS)
        {
            address += context.GsBase;
        }
        return true;
    }

    /// <summary>
    /// Reads guest memory, falling back to a direct host-pointer read when the address
    /// is not part of the guest-mapped address space. HLE exports such as malloc return
    /// raw host allocations to guest code, which the native backend can dereference
    /// directly (it executes in the same process); the interpreter needs the same
    /// fallback to stay compatible.
    /// </summary>
    private static bool TryReadInterpreterMemory(CpuContext context, ulong address, Span<byte> destination) =>
        context.Memory.TryRead(address, destination) ||
        KernelMemoryCompatExports.TryReadRawHostMemory(address, destination);

    /// <summary>See <see cref="TryReadInterpreterMemory"/>.</summary>
    private static bool TryWriteInterpreterMemory(CpuContext context, ulong address, ReadOnlySpan<byte> source) =>
        context.Memory.TryWrite(address, source) ||
        KernelMemoryCompatExports.TryWriteRawHostMemory(address, source);

    private static bool TryReadMemory(
        CpuContext context,
        ulong address,
        int bits,
        out ulong value,
        out InterpreterFailure failure)
    {
        Span<byte> bytes = stackalloc byte[8];
        var size = bits / 8;
        if (!TryReadInterpreterMemory(context, address, bytes[..size]))
        {
            value = 0;
            failure = InterpreterFailure.MemoryRead(address, size);
            return false;
        }
        value = bits switch
        {
            8 => bytes[0],
            16 => BinaryPrimitives.ReadUInt16LittleEndian(bytes),
            32 => BinaryPrimitives.ReadUInt32LittleEndian(bytes),
            64 => BinaryPrimitives.ReadUInt64LittleEndian(bytes),
            _ => 0,
        };
        failure = default;
        return bits is 8 or 16 or 32 or 64;
    }

    private static bool TryWriteMemory(
        CpuContext context,
        ulong address,
        ulong value,
        int bits,
        out InterpreterFailure failure)
    {
        Span<byte> bytes = stackalloc byte[8];
        var size = bits / 8;
        switch (bits)
        {
            case 8:
                bytes[0] = (byte)value;
                break;
            case 16:
                BinaryPrimitives.WriteUInt16LittleEndian(bytes, (ushort)value);
                break;
            case 32:
                BinaryPrimitives.WriteUInt32LittleEndian(bytes, (uint)value);
                break;
            case 64:
                BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
                break;
            default:
                failure = InterpreterFailure.Unsupported($"unsupported memory write size {bits}");
                return false;
        }
        if (!TryWriteInterpreterMemory(context, address, bytes[..size]))
        {
            failure = InterpreterFailure.MemoryWrite(address, size);
            return false;
        }
        failure = default;
        return true;
    }

    private static bool TryReadRegister(
        CpuContext context,
        Register register,
        int bits,
        out ulong value,
        out InterpreterFailure failure)
    {
        if (!TryGetRegisterInfo(register, out var cpuRegister, out var registerBits, out var bitOffset))
        {
            value = 0;
            failure = InterpreterFailure.Unsupported($"unsupported register {register}");
            return false;
        }

        var raw = context[cpuRegister];
        var effectiveBits = Math.Min(bits, registerBits);
        value = (raw >> bitOffset) & X64Flags.Mask(effectiveBits);
        failure = default;
        return true;
    }

    private static bool TryWriteRegister(
        CpuContext context,
        Register register,
        ulong value,
        int bits,
        out InterpreterFailure failure)
    {
        if (!TryGetRegisterInfo(register, out var cpuRegister, out var registerBits, out var bitOffset))
        {
            failure = InterpreterFailure.Unsupported($"unsupported register {register}");
            return false;
        }

        var effectiveBits = Math.Min(bits, registerBits);
        value &= X64Flags.Mask(effectiveBits);
        if (registerBits == 32 && bitOffset == 0)
        {
            context[cpuRegister] = value;
        }
        else if (registerBits == 64 && bitOffset == 0)
        {
            context[cpuRegister] = value;
        }
        else
        {
            var clearMask = ~(X64Flags.Mask(effectiveBits) << bitOffset);
            context[cpuRegister] = (context[cpuRegister] & clearMask) | (value << bitOffset);
        }

        failure = default;
        return true;
    }

    private static bool TryGetBaseRegisterValue(CpuContext context, Register register, out ulong value)
    {
        if (register == Register.None)
        {
            value = 0;
            return true;
        }
        if (!TryGetRegisterInfo(register, out var cpuRegister, out _, out _))
        {
            value = 0;
            return false;
        }
        value = context[cpuRegister];
        return true;
    }

    private static bool TryGetRegisterInfo(
        Register register,
        out CpuRegister cpuRegister,
        out int bits,
        out int bitOffset)
    {
        bitOffset = 0;
        (cpuRegister, bits) = register switch
        {
            Register.RAX => (CpuRegister.Rax, 64),
            Register.RCX => (CpuRegister.Rcx, 64),
            Register.RDX => (CpuRegister.Rdx, 64),
            Register.RBX => (CpuRegister.Rbx, 64),
            Register.RSP => (CpuRegister.Rsp, 64),
            Register.RBP => (CpuRegister.Rbp, 64),
            Register.RSI => (CpuRegister.Rsi, 64),
            Register.RDI => (CpuRegister.Rdi, 64),
            Register.R8 => (CpuRegister.R8, 64),
            Register.R9 => (CpuRegister.R9, 64),
            Register.R10 => (CpuRegister.R10, 64),
            Register.R11 => (CpuRegister.R11, 64),
            Register.R12 => (CpuRegister.R12, 64),
            Register.R13 => (CpuRegister.R13, 64),
            Register.R14 => (CpuRegister.R14, 64),
            Register.R15 => (CpuRegister.R15, 64),
            Register.EAX => (CpuRegister.Rax, 32),
            Register.ECX => (CpuRegister.Rcx, 32),
            Register.EDX => (CpuRegister.Rdx, 32),
            Register.EBX => (CpuRegister.Rbx, 32),
            Register.ESP => (CpuRegister.Rsp, 32),
            Register.EBP => (CpuRegister.Rbp, 32),
            Register.ESI => (CpuRegister.Rsi, 32),
            Register.EDI => (CpuRegister.Rdi, 32),
            Register.R8D => (CpuRegister.R8, 32),
            Register.R9D => (CpuRegister.R9, 32),
            Register.R10D => (CpuRegister.R10, 32),
            Register.R11D => (CpuRegister.R11, 32),
            Register.R12D => (CpuRegister.R12, 32),
            Register.R13D => (CpuRegister.R13, 32),
            Register.R14D => (CpuRegister.R14, 32),
            Register.R15D => (CpuRegister.R15, 32),
            Register.AX => (CpuRegister.Rax, 16),
            Register.CX => (CpuRegister.Rcx, 16),
            Register.DX => (CpuRegister.Rdx, 16),
            Register.BX => (CpuRegister.Rbx, 16),
            Register.SP => (CpuRegister.Rsp, 16),
            Register.BP => (CpuRegister.Rbp, 16),
            Register.SI => (CpuRegister.Rsi, 16),
            Register.DI => (CpuRegister.Rdi, 16),
            Register.R8W => (CpuRegister.R8, 16),
            Register.R9W => (CpuRegister.R9, 16),
            Register.R10W => (CpuRegister.R10, 16),
            Register.R11W => (CpuRegister.R11, 16),
            Register.R12W => (CpuRegister.R12, 16),
            Register.R13W => (CpuRegister.R13, 16),
            Register.R14W => (CpuRegister.R14, 16),
            Register.R15W => (CpuRegister.R15, 16),
            Register.AL => (CpuRegister.Rax, 8),
            Register.CL => (CpuRegister.Rcx, 8),
            Register.DL => (CpuRegister.Rdx, 8),
            Register.BL => (CpuRegister.Rbx, 8),
            Register.SPL => (CpuRegister.Rsp, 8),
            Register.BPL => (CpuRegister.Rbp, 8),
            Register.SIL => (CpuRegister.Rsi, 8),
            Register.DIL => (CpuRegister.Rdi, 8),
            Register.R8L => (CpuRegister.R8, 8),
            Register.R9L => (CpuRegister.R9, 8),
            Register.R10L => (CpuRegister.R10, 8),
            Register.R11L => (CpuRegister.R11, 8),
            Register.R12L => (CpuRegister.R12, 8),
            Register.R13L => (CpuRegister.R13, 8),
            Register.R14L => (CpuRegister.R14, 8),
            Register.R15L => (CpuRegister.R15, 8),
            Register.AH => (CpuRegister.Rax, 8),
            Register.CH => (CpuRegister.Rcx, 8),
            Register.DH => (CpuRegister.Rdx, 8),
            Register.BH => (CpuRegister.Rbx, 8),
            _ => ((CpuRegister)(-1), 0),
        };

        if (register is Register.AH or Register.CH or Register.DH or Register.BH)
        {
            bitOffset = 8;
        }

        return bits != 0;
    }

    private static bool TryGetVectorRegisterInfo(Register register, out int index, out int size)
    {
        index = register switch
        {
            Register.XMM0 or Register.YMM0 => 0,
            Register.XMM1 or Register.YMM1 => 1,
            Register.XMM2 or Register.YMM2 => 2,
            Register.XMM3 or Register.YMM3 => 3,
            Register.XMM4 or Register.YMM4 => 4,
            Register.XMM5 or Register.YMM5 => 5,
            Register.XMM6 or Register.YMM6 => 6,
            Register.XMM7 or Register.YMM7 => 7,
            Register.XMM8 or Register.YMM8 => 8,
            Register.XMM9 or Register.YMM9 => 9,
            Register.XMM10 or Register.YMM10 => 10,
            Register.XMM11 or Register.YMM11 => 11,
            Register.XMM12 or Register.YMM12 => 12,
            Register.XMM13 or Register.YMM13 => 13,
            Register.XMM14 or Register.YMM14 => 14,
            Register.XMM15 or Register.YMM15 => 15,
            _ => -1,
        };
        size = register.ToString().StartsWith("YMM", StringComparison.Ordinal) ? 32 : 16;
        return index >= 0;
    }

    private static bool TryReadVectorOperand(
        CpuContext context,
        in Instruction instruction,
        int opIndex,
        int size,
        Span<byte> destination,
        out InterpreterFailure failure)
    {
        switch (instruction.GetOpKind(opIndex))
        {
            case OpKind.Register:
                if (!TryGetVectorRegisterInfo(instruction.GetOpRegister(opIndex), out var registerIndex, out _))
                {
                    failure = InterpreterFailure.Unsupported($"unsupported vector register {instruction.GetOpRegister(opIndex)}");
                    return false;
                }

                GetVectorRegister(context, registerIndex, size, destination);
                failure = default;
                return true;
            case OpKind.Memory:
                if (!TryGetMemoryAddress(context, instruction, out var address))
                {
                    failure = InterpreterFailure.Unsupported("unsupported vector memory address form");
                    return false;
                }

                if (!TryReadInterpreterMemory(context, address, destination))
                {
                    failure = InterpreterFailure.MemoryRead(address, size);
                    return false;
                }

                failure = default;
                return true;
            default:
                failure = InterpreterFailure.Unsupported($"unsupported vector operand kind {instruction.GetOpKind(opIndex)}");
                return false;
        }
    }

    private static void GetVectorRegister(CpuContext context, int index, int size, Span<byte> destination)
    {
        if (size == 32)
        {
            context.GetYmmRegister(index, out var lowLow, out var lowHigh, out var highLow, out var highHigh);
            BinaryPrimitives.WriteUInt64LittleEndian(destination, lowLow);
            BinaryPrimitives.WriteUInt64LittleEndian(destination[8..], lowHigh);
            BinaryPrimitives.WriteUInt64LittleEndian(destination[16..], highLow);
            BinaryPrimitives.WriteUInt64LittleEndian(destination[24..], highHigh);
            return;
        }

        context.GetXmmRegister(index, out var low, out var high);
        BinaryPrimitives.WriteUInt64LittleEndian(destination, low);
        BinaryPrimitives.WriteUInt64LittleEndian(destination[8..], high);
    }

    private static void SetVectorRegister(CpuContext context, int index, ReadOnlySpan<byte> source)
    {
        var lowLow = BinaryPrimitives.ReadUInt64LittleEndian(source);
        var lowHigh = BinaryPrimitives.ReadUInt64LittleEndian(source[8..]);
        if (source.Length == 32)
        {
            var highLow = BinaryPrimitives.ReadUInt64LittleEndian(source[16..]);
            var highHigh = BinaryPrimitives.ReadUInt64LittleEndian(source[24..]);
            context.SetYmmRegister(index, lowLow, lowHigh, highLow, highHigh);
            return;
        }

        context.SetXmmRegister(index, lowLow, lowHigh);
        context.ClearYmmUpper(index);
    }

    private static int GetOperandBitSize(in Instruction instruction, int opIndex)
    {
        var size = instruction.GetOpKind(opIndex) == OpKind.Immediate8to64
            ? 64
            : instruction.GetOpKind(opIndex) == OpKind.Immediate8to32
                ? 32
                : instruction.GetOpKind(opIndex) == OpKind.Immediate8to16
                    ? 16
                    : instruction.GetOpKind(opIndex) == OpKind.Immediate32to64
                        ? 64
                        : instruction.GetOpKind(opIndex) == OpKind.Immediate8
                            ? 8
                            : instruction.GetOpKind(opIndex) == OpKind.Immediate16
                                ? 16
                                : instruction.GetOpKind(opIndex) == OpKind.Immediate32
                                    ? 32
                                    : instruction.GetOpKind(opIndex) == OpKind.Immediate64
                                        ? 64
                                        : instruction.GetOpKind(opIndex) == OpKind.Register
                                            ? GetRegisterBitSize(instruction.GetOpRegister(opIndex))
                                            : instruction.MemorySize.GetSize() * 8;
        return size is 8 or 16 or 32 or 64 ? size : 64;
    }

    private static int GetRegisterBitSize(Register register)
    {
        _ = TryGetRegisterInfo(register, out _, out var bits, out _);
        return bits == 0 ? 64 : bits;
    }

    private static bool Unsupported(in Instruction instruction, out InterpreterFailure failure)
    {
        failure = InterpreterFailure.Unsupported($"instruction is not implemented: {instruction}");
        return false;
    }

    private X64InterpreterResult NotImplemented(
        ulong rip,
        string mnemonic,
        string detail,
        int totalInstructions,
        int importsHit,
        int uniqueNidsHit,
        StringBuilder? trace)
    {
        return Complete(
            OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_IMPLEMENTED,
            CpuExitReason.NativeBackendUnavailable,
            rip,
            totalInstructions,
            importsHit,
            uniqueNidsHit,
            trace,
            notImplementedInfo: new CpuNotImplementedInfo(
                CpuNotImplementedSource.Interpreter,
                rip,
                nid: null,
                exportName: mnemonic,
                libraryName: "x64-interpreter",
                detail: detail));
    }

    private X64InterpreterResult Trap(
        ulong rip,
        byte opcode,
        int totalInstructions,
        int importsHit,
        int uniqueNidsHit,
        StringBuilder? trace)
    {
        return Complete(
            OrbisGen2Result.ORBIS_GEN2_ERROR_CPU_TRAP,
            CpuExitReason.CpuTrap,
            rip,
            totalInstructions,
            importsHit,
            uniqueNidsHit,
            trace,
            trapInfo: new CpuTrapInfo(rip, opcode));
    }

    private X64InterpreterResult MemoryFault(
        CpuContext context,
        ulong rip,
        byte? opcode,
        ulong address,
        int size,
        bool isWrite,
        int totalInstructions,
        int importsHit,
        int uniqueNidsHit,
        StringBuilder? trace)
    {
        if (trace is not null)
        {
            var accessType = isWrite ? "write" : "read";
            trace.AppendLine(
                $"[CPU-INTERP][FAULT] rip=0x{rip:X16} {accessType}@0x{address:X16} size={size} {FormatRegisters(context)}");
        }

        return Complete(
            OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT,
            CpuExitReason.UnhandledException,
            rip,
            totalInstructions,
            importsHit,
            uniqueNidsHit,
            trace,
            memoryFaultInfo: new CpuMemoryFaultInfo(rip, opcode, new CpuMemoryAccessFailure(address, size, isWrite)));
    }

    private X64InterpreterResult Complete(
        OrbisGen2Result result,
        CpuExitReason reason,
        ulong lastGuestRip,
        int totalInstructions,
        int importsHit,
        int uniqueNidsHit,
        StringBuilder? trace,
        CpuTrapInfo? trapInfo = null,
        CpuMemoryFaultInfo? memoryFaultInfo = null,
        CpuNotImplementedInfo? notImplementedInfo = null)
    {
        return new X64InterpreterResult(
            result,
            reason,
            lastGuestRip,
            totalInstructions,
            importsHit,
            uniqueNidsHit,
            trapInfo,
            memoryFaultInfo,
            notImplementedInfo,
            trace?.ToString(),
            FormatRecentInstructions());
    }

    private string FormatRecentInstructions()
    {
        if (_recentInstructionCount == 0)
        {
            return string.Empty;
        }

        // When the buffer hasn't wrapped yet, the oldest entry is at index 0; once full, it's
        // whatever PushRecent is about to overwrite next (_recentInstructionHead).
        var startIndex = _recentInstructionCount < RecentInstructionCapacity ? 0 : _recentInstructionHead;
        var lines = new string[_recentInstructionCount];
        for (var i = 0; i < _recentInstructionCount; i++)
        {
            var entry = _recentInstructions[(startIndex + i) & (RecentInstructionCapacity - 1)];
            lines[i] = FormatInstruction(entry.Instruction, entry.Bytes);
        }

        return string.Join(Environment.NewLine, lines);
    }

    private const int RecentInstructionCapacity = 256;

    private void PushRecent(in Instruction instruction, byte[] bytes)
    {
        _recentInstructions[_recentInstructionHead] = (instruction, bytes);
        _recentInstructionHead = (_recentInstructionHead + 1) & (RecentInstructionCapacity - 1);
        if (_recentInstructionCount < RecentInstructionCapacity)
        {
            _recentInstructionCount++;
        }
    }

    private static string FormatInstruction(in Instruction instruction, ReadOnlySpan<byte> bytes)
    {
        var formatter = new IntelFormatter();
        var output = new StringOutput();
        formatter.Format(instruction, output);
        return $"[CPU-INTERP] rip=0x{instruction.IP:X16} bytes={FormatBytes(bytes[..instruction.Length])} inst=\"{output}\"";
    }

    private static string FormatBytes(ReadOnlySpan<byte> bytes)
    {
        if (bytes.IsEmpty)
        {
            return "??";
        }
        var parts = new string[bytes.Length];
        for (var i = 0; i < bytes.Length; i++)
        {
            parts[i] = bytes[i].ToString("X2");
        }
        return string.Join(' ', parts);
    }

    private static string FormatRegisters(CpuContext context) =>
        $"rax=0x{context[CpuRegister.Rax]:X16} rbx=0x{context[CpuRegister.Rbx]:X16} " +
        $"rcx=0x{context[CpuRegister.Rcx]:X16} rdx=0x{context[CpuRegister.Rdx]:X16} " +
        $"rsi=0x{context[CpuRegister.Rsi]:X16} rdi=0x{context[CpuRegister.Rdi]:X16} " +
        $"rsp=0x{context[CpuRegister.Rsp]:X16} rbp=0x{context[CpuRegister.Rbp]:X16} " +
        $"r8=0x{context[CpuRegister.R8]:X16} r9=0x{context[CpuRegister.R9]:X16} " +
        $"r10=0x{context[CpuRegister.R10]:X16} r11=0x{context[CpuRegister.R11]:X16} " +
        $"r12=0x{context[CpuRegister.R12]:X16} r13=0x{context[CpuRegister.R13]:X16} " +
        $"r14=0x{context[CpuRegister.R14]:X16} r15=0x{context[CpuRegister.R15]:X16} " +
        $"rflags=0x{context.Rflags:X16}";

    private enum BinaryOp
    {
        Add,
        Sub,
        Xor,
        And,
        Or,
    }

    private enum UnaryOp
    {
        Inc,
        Dec,
        Neg,
        Not,
    }

    private enum ShiftOp
    {
        Shl,
        Shr,
        Sar,
    }

    private enum ShiftxOp
    {
        Shlx,
        Shrx,
        Sarx,
    }

    private enum VariableShiftOp
    {
        Srlv,
        Sllv,
        Srav,
    }

    private enum VectorBitwiseOp
    {
        And,
        Or,
        Xor,
        AndNot,
    }

    private enum PackedArithOp
    {
        Add,
        Sub,
    }

    private enum UnpackHalf
    {
        Low,
        High,
    }

    private enum PackedShiftOp
    {
        Sll,
        Srl,
        Sra,
    }

    private enum ScalarArithOp
    {
        Add,
        Sub,
        Mul,
        Div,
        Min,
        Max,
    }

    private readonly struct InterpreterFailure
    {
        private InterpreterFailure(InterpreterFailureKind kind, ulong address, int size, string detail)
        {
            Kind = kind;
            Address = address;
            Size = size;
            Detail = detail;
        }

        public InterpreterFailureKind Kind { get; }

        public ulong Address { get; }

        public int Size { get; }

        public string Detail { get; }

        public static InterpreterFailure Unsupported(string detail) =>
            new(InterpreterFailureKind.Unsupported, 0, 0, detail);

        public static InterpreterFailure MemoryRead(ulong address, int size) =>
            new(InterpreterFailureKind.MemoryRead, address, size, string.Empty);

        public static InterpreterFailure MemoryWrite(ulong address, int size) =>
            new(InterpreterFailureKind.MemoryWrite, address, size, string.Empty);

        public static InterpreterFailure Trap() =>
            new(InterpreterFailureKind.Trap, 0, 0, string.Empty);
    }

    private enum InterpreterFailureKind
    {
        Unsupported,
        MemoryRead,
        MemoryWrite,
        Trap,
    }
}
