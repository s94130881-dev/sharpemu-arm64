// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text;
using System.Threading;
using SharpEmu.Core.Cpu.Debugging;
using SharpEmu.Core.Cpu.Interpreter;
using SharpEmu.Core.Cpu.Native;
using SharpEmu.Core.Loader;
using SharpEmu.Core.Memory;
using SharpEmu.HLE;

namespace SharpEmu.Core.Cpu;

/// <summary>
/// Creates and dispatches a guest x86-64 CPU context.
///
/// Android/ARM64 deliberately uses the managed x86-64 interpreter.
/// The native backend is reserved for hosts that can execute the guest
/// architecture directly.
///
/// The addresses used here are GUEST virtual addresses. They do not represent
/// physical RAM and increasing them does not create additional RAM.
/// </summary>
public sealed class CpuDispatcher : ICpuDispatcher, IDisposable
{
    private enum EntryFrameKind
    {
        ProcessEntry,
        ModuleInitializer
    }

    /*
     * -----------------------------------------------------------------------
     * Android guest address layout
     * -----------------------------------------------------------------------
     *
     * Keep these areas:
     *
     *   - inside a conservative 39-bit VA range
     *   - separated from one another
     *   - below the normal guest mmap search area
     *   - independent from the physical amount of RAM in the phone
     *
     * These are guest virtual addresses, not host RAM allocations.
     */

    private static readonly bool IsAndroid = OperatingSystem.IsAndroid();

    private const ulong AndroidStackBase = 0x0000_000C_0000_0000UL;
    private const ulong AndroidTlsBase = 0x0000_000D_0000_0000UL;

    private const ulong AndroidBootstrapBase = 0x0000_000E_0000_0000UL;
    private const ulong AndroidPayloadBase = 0x0000_000E_4000_0000UL;
    private const ulong AndroidDynlibStubBase = 0x0000_000E_8000_0000UL;
    private const ulong AndroidReturnStubBase = 0x0000_000E_C000_0000UL;

    private const ulong WindowsStackBase = 0x0000_7FFF_F000_0000UL;
    private const ulong WindowsTlsBase = 0x0000_7FFE_0000_0000UL;
    private const ulong WindowsBootstrapBase = 0x0000_7FFD_F000_0000UL;
    private const ulong WindowsPayloadBase = 0x0000_7FFD_E000_0000UL;
    private const ulong WindowsDynlibStubBase = 0x0000_7FFD_D000_0000UL;
    private const ulong WindowsReturnStubBase = 0x0000_7FFD_C000_0000UL;

    private const ulong PosixStackBase = 0x0000_6FFF_F000_0000UL;
    private const ulong PosixTlsBase = 0x0000_6FFE_0000_0000UL;
    private const ulong PosixBootstrapBase = 0x0000_6FFD_F000_0000UL;
    private const ulong PosixPayloadBase = 0x0000_6FFD_E000_0000UL;
    private const ulong PosixDynlibStubBase = 0x0000_6FFD_D000_0000UL;
    private const ulong PosixReturnStubBase = 0x0000_6FFD_C000_0000UL;

    private static ulong StackBaseAddress =>
        IsAndroid
            ? AndroidStackBase
            : OperatingSystem.IsWindows()
                ? WindowsStackBase
                : PosixStackBase;

    private static ulong TlsBaseAddress =>
        IsAndroid
            ? AndroidTlsBase
            : OperatingSystem.IsWindows()
                ? WindowsTlsBase
                : PosixTlsBase;

    private static ulong BootstrapStubBaseAddress =>
        IsAndroid
            ? AndroidBootstrapBase
            : OperatingSystem.IsWindows()
                ? WindowsBootstrapBase
                : PosixBootstrapBase;

    private static ulong BootstrapPayloadBaseAddress =>
        IsAndroid
            ? AndroidPayloadBase
            : OperatingSystem.IsWindows()
                ? WindowsPayloadBase
                : PosixPayloadBase;

    private static ulong DynlibFallbackStubBaseAddress =>
        IsAndroid
            ? AndroidDynlibStubBase
            : OperatingSystem.IsWindows()
                ? WindowsDynlibStubBase
                : PosixDynlibStubBase;

    private static ulong ReturnToHostStubBaseAddress =>
        IsAndroid
            ? AndroidReturnStubBase
            : OperatingSystem.IsWindows()
                ? WindowsReturnStubBase
                : PosixReturnStubBase;

    /*
     * 2 MiB guest stack.
     *
     * This is NOT 2 MiB of host RAM necessarily. The actual host allocation
     * depends on IVirtualMemory implementation.
     */
    internal const ulong StackSize = 0x0020_0000UL;

    private const ulong TlsSize = 0x0001_0000UL;

    private const ulong StackSlotStride = 0x0100_0000UL;
    private const ulong TlsSlotStride = 0x0100_0000UL;

    private const int MaxStackSlots = 32;
    private const int MaxTlsSlots = 32;

    private const int MaxStubSlots = 16;

    private const ulong StubRegionSize = 0x1000UL;
    private const ulong StubStride = 0x0100_0000UL;

    private const ulong BootstrapPayloadResultOffset = 0x28UL;
    private const ulong BootstrapStatusOffset = 0x100UL;

    private static readonly byte[] BootstrapStartSignature =
    [
        0x55, 0x48, 0x89, 0xE5,
        0x41, 0x57,
        0x41, 0x56,
        0x41, 0x55,
        0x41, 0x54,
        0x53,
        0x50,
        0x48, 0x89
    ];

    private readonly IVirtualMemory _virtualMemory;
    private readonly IModuleManager _moduleManager;

    private INativeCpuBackend? _nativeCpuBackend;

    private int _nextStackSlot = -1;
    private int _nextTlsSlot = -1;

    internal bool NativeSessionLeaked { get; private set; }

    public CpuDispatcher(
        IVirtualMemory virtualMemory,
        IModuleManager moduleManager,
        INativeCpuBackend? nativeCpuBackend = null)
    {
        _virtualMemory =
            virtualMemory ??
            throw new ArgumentNullException(nameof(virtualMemory));

        _moduleManager =
            moduleManager ??
            throw new ArgumentNullException(nameof(moduleManager));

        _nativeCpuBackend = nativeCpuBackend;
    }

    public ulong? LastEntryPoint { get; private set; }

    public CpuTrapInfo? LastTrapInfo { get; private set; }

    public CpuMemoryFaultInfo? LastMemoryFaultInfo { get; private set; }

    public CpuControlTransferInfo? LastControlTransferInfo { get; private set; }

    public CpuNotImplementedInfo? LastNotImplementedInfo { get; private set; }

    public string? LastImportResolutionTrace { get; private set; }

    public string? LastBasicBlockTrace { get; private set; }

    public string? LastMilestoneLog { get; private set; }

    public string? LastRecentInstructionWindow { get; private set; }

    public string? LastRecentControlTransferTrace { get; private set; }

    public CpuSessionSummary LastSessionSummary { get; private set; }

    // ---------------------------------------------------------------------
    // Public dispatch
    // ---------------------------------------------------------------------

    public OrbisGen2Result DispatchEntry(
        ulong entryPoint,
        Generation generation,
        IReadOnlyDictionary<ulong, string>? importStubs = null,
        IReadOnlyDictionary<string, ulong>? runtimeSymbols = null,
        string processImageName = "eboot.bin",
        CpuExecutionOptions executionOptions = default)
    {
        Console.Error.WriteLine(
            $"[DISPATCHER] START entry=0x{entryPoint:X16} generation={generation} android={IsAndroid}");

        try
        {
            return DispatchEntryCore(
                entryPoint,
                generation,
                importStubs,
                runtimeSymbols,
                processImageName,
                executionOptions,
                EntryFrameKind.ProcessEntry);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"[DISPATCHER] FATAL {ex.GetType().Name}: {ex.Message}");

            Console.Error.WriteLine(ex.StackTrace);

            throw;
        }
    }

    public OrbisGen2Result DispatchModuleInitializer(
        ulong entryPoint,
        Generation generation,
        IReadOnlyDictionary<ulong, string>? importStubs = null,
        IReadOnlyDictionary<string, ulong>? runtimeSymbols = null,
        string moduleName = "module",
        CpuExecutionOptions executionOptions = default)
    {
        Console.Error.WriteLine(
            $"[DISPATCHER] START module=0x{entryPoint:X16} generation={generation} name={moduleName}");

        try
        {
            return DispatchEntryCore(
                entryPoint,
                generation,
                importStubs,
                runtimeSymbols,
                moduleName,
                executionOptions,
                EntryFrameKind.ModuleInitializer);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"[DISPATCHER] FATAL {ex.GetType().Name}: {ex.Message}");

            Console.Error.WriteLine(ex.StackTrace);

            throw;
        }
    }

    // ---------------------------------------------------------------------
    // Core dispatcher
    // ---------------------------------------------------------------------

    private OrbisGen2Result DispatchEntryCore(
        ulong entryPoint,
        Generation generation,
        IReadOnlyDictionary<ulong, string>? importStubs,
        IReadOnlyDictionary<string, ulong>? runtimeSymbols,
        string processImageName,
        CpuExecutionOptions executionOptions,
        EntryFrameKind frameKind)
    {
        ResetDiagnostics();

        LastEntryPoint = entryPoint;

        if (entryPoint == 0)
        {
            LastNotImplementedInfo = new CpuNotImplementedInfo(
                CpuNotImplementedSource.NativeBackend,
                0,
                null,
                "invalid_entry_point",
                "dispatcher",
                "Guest entry point is zero.");

            return FailEarly(
                OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT,
                CpuExitReason.UnhandledException);
        }

        /*
         * Android MUST use the interpreter.
         *
         * Do not silently fall back to a native x86-64 backend.
         */
        if (IsAndroid &&
            executionOptions.CpuEngine != CpuExecutionEngine.Interpreter)
        {
            executionOptions = ForceInterpreter(executionOptions);

            Console.Error.WriteLine(
                "[DISPATCHER] Android/ARM64: forcing X64 interpreter.");
        }

        var stackBase = TryMapStackRegion();

        if (stackBase == 0)
        {
            return FailEarly(
                OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        var tlsBase = TryMapTlsRegion();

        if (tlsBase == 0)
        {
            return FailEarly(
                OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        var trackedMemory = new TrackedCpuMemory(_virtualMemory);

        var context = new CpuContext(
            trackedMemory,
            generation)
        {
            Rip = entryPoint,
            Rflags = 0x202,
            FsBase = tlsBase,
            GsBase = tlsBase
        };

        /*
         * Reserve the return-to-host location before constructing the
         * initial stack.
         */
        var returnStub = TryMapReturnToHostStubRegion();

        if (returnStub == 0)
        {
            return FailEarly(
                OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        context[CpuRegister.Rsp] =
            stackBase +
            StackSize -
            sizeof(ulong);

        if (!context.TryWriteUInt64(
                context[CpuRegister.Rsp],
                returnStub))
        {
            return FailEarly(
                OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        if (!InitializeGuestFrameChainSentinel(context))
        {
            return FailEarly(
                OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        if (!InitializeTls(context, tlsBase))
        {
            return FailEarly(
                OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
        }

        var effectiveImportStubs =
            importStubs is null
                ? new Dictionary<ulong, string>()
                : new Dictionary<ulong, string>(importStubs);

        var entryParamsConfigured = false;

        if (frameKind == EntryFrameKind.ProcessEntry)
        {
            var exitStub =
                TryMapDynlibFallbackStubRegion();

            if (exitStub == 0)
            {
                return FailEarly(
                    OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }

            if (!InitializeProcessEntryFrame(
                    context,
                    processImageName,
                    exitStub))
            {
                return FailEarly(
                    OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }

            entryParamsConfigured = true;

            /*
             * Bootstrap injection is intentionally conservative.
             * It is only installed when the entry-point signature matches.
             */
            if (ShouldInjectBootstrapPayload(entryPoint))
            {
                Console.Error.WriteLine(
                    "[DISPATCHER] Bootstrap signature detected.");

                if (!TryInstallBootstrapPayload(
                        context,
                        effectiveImportStubs))
                {
                    return FailEarly(
                        OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
                }
            }
        }
        else
        {
            if (!InitializeModuleInitializerFrame(context))
            {
                return FailEarly(
                    OrbisGen2Result.ORBIS_GEN2_ERROR_MEMORY_FAULT);
            }
        }

        LastMilestoneLog = BuildEntryFrameDiagnostic(
            entryPoint,
            context,
            true,
            returnStub,
            entryParamsConfigured);

        /*
         * ---------------------------------------------------------------
         * Android / ARM64 interpreter
         * ---------------------------------------------------------------
         */
        if (executionOptions.CpuEngine ==
            CpuExecutionEngine.Interpreter)
        {
            return ExecuteInterpreter(
                context,
                entryPoint,
                effectiveImportStubs,
                executionOptions);
        }

        /*
         * ---------------------------------------------------------------
         * Native backend
         * ---------------------------------------------------------------
         *
         * Android has already been forced to Interpreter above.
         * This path is for desktop/native execution only.
         */
        if (executionOptions.CpuEngine !=
            CpuExecutionEngine.NativeOnly)
        {
            LastNotImplementedInfo = new CpuNotImplementedInfo(
                CpuNotImplementedSource.NativeBackend,
                entryPoint,
                null,
                "cpu_engine_unsupported",
                executionOptions.CpuEngine.ToString(),
                "Unsupported CPU engine mode.");

            return FailEarly(
                OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_IMPLEMENTED,
                CpuExitReason.NativeBackendUnavailable);
        }

        return ExecuteNative(
            context,
            entryPoint,
            generation,
            effectiveImportStubs,
            runtimeSymbols,
            processImageName,
            executionOptions,
            frameKind);
    }

    // ---------------------------------------------------------------------
    // Interpreter
    // ---------------------------------------------------------------------

    private OrbisGen2Result ExecuteInterpreter(
        CpuContext context,
        ulong entryPoint,
        IReadOnlyDictionary<ulong, string> importStubs,
        CpuExecutionOptions executionOptions)
    {
        LastMilestoneLog = string.Concat(
            LastMilestoneLog,
            Environment.NewLine,
            $"CpuEngine: x64-interpreter",
            Environment.NewLine,
            $"Trace: {executionOptions.InterpreterTrace}",
            Environment.NewLine,
            $"MaxInstructions: {executionOptions.EffectiveInterpreterMaxInstructions}");

        var options = new X64InterpreterOptions
        {
            Trace = executionOptions.InterpreterTrace,
            MaxInstructions =
                executionOptions.EffectiveInterpreterMaxInstructions
        };

        var scheduler =
            new X64InterpreterGuestThreadScheduler(
                this,
                _moduleManager,
                importStubs,
                options);

        var previousScheduler =
            GuestThreadExecution.Scheduler;

        GuestThreadExecution.Scheduler =
            scheduler;

        try
        {
            var interpreter =
                new X64InterpreterBackend(
                    _moduleManager,
                    scheduler);

            var result =
                interpreter.Execute(
                    context,
                    entryPoint,
                    importStubs,
                    options);

            LastTrapInfo = result.TrapInfo;
            LastMemoryFaultInfo = result.MemoryFaultInfo;
            LastNotImplementedInfo = result.NotImplementedInfo;

            LastBasicBlockTrace = result.Trace;
            LastRecentInstructionWindow = result.RecentInstructions;

            LastSessionSummary =
                new CpuSessionSummary(
                    result.Result,
                    result.Reason,
                    exitCode: null,
                    lastGuestRip: result.LastGuestRip,
                    lastStubRip: 0,
                    totalInstructions: result.TotalInstructions,
                    importsHit: result.ImportsHit,
                    uniqueNidsHit: result.UniqueNidsHit);

            Console.Error.WriteLine(
                $"[DISPATCHER] Interpreter finished: result={result.Result} reason={result.Reason} rip=0x{result.LastGuestRip:X16} instructions={result.TotalInstructions}");

            return result.Result;
        }
        finally
        {
            GuestThreadExecution.Scheduler =
                previousScheduler;
        }
    }

    // ---------------------------------------------------------------------
    // Native
    // ---------------------------------------------------------------------

    private OrbisGen2Result ExecuteNative(
        CpuContext context,
        ulong entryPoint,
        Generation generation,
        IReadOnlyDictionary<ulong, string> importStubs,
        IReadOnlyDictionary<string, ulong>? runtimeSymbols,
        string processImageName,
        CpuExecutionOptions executionOptions,
        EntryFrameKind frameKind)
    {
        /*
         * Safety check.
         */
        if (IsAndroid)
        {
            throw new PlatformNotSupportedException(
                "Native x86-64 guest execution is not supported on Android/ARM64. Use the X64 interpreter.");
        }

        var debugHook =
            executionOptions.DebugHook;

        var debugFrame =
            debugHook is null
                ? null
                : new CpuContextDebugFrame(
                    frameKind == EntryFrameKind.ProcessEntry
                        ? CpuDebugFrameKind.ProcessEntry
                        : CpuDebugFrameKind.ModuleInitializer,
                    entryPoint,
                    processImageName,
                    context,
                    importStubs);

        debugHook?.OnFrameEnter(debugFrame!);

        _nativeCpuBackend ??=
            new DirectExecutionBackend(
                _moduleManager);

        (
            _nativeCpuBackend as DirectExecutionBackend
        )?.SetActiveDebugFrame(debugFrame);

        if (_nativeCpuBackend.TryExecute(
                context,
                entryPoint,
                generation,
                importStubs,
                runtimeSymbols ??
                new Dictionary<string, ulong>(
                    StringComparer.Ordinal),
                executionOptions,
                out var nativeResult))
        {
            debugHook?.OnFrameExit(
                debugFrame!,
                nativeResult);

            LastSessionSummary =
                new CpuSessionSummary(
                    nativeResult,
                    nativeResult ==
                    OrbisGen2Result.ORBIS_GEN2_OK
                        ? CpuExitReason.ReturnedToHost
                        : CpuExitReason.UnhandledException,
                    exitCode: null,
                    lastGuestRip: context.Rip,
                    lastStubRip: 0,
                    totalInstructions: 0,
                    importsHit: 0,
                    uniqueNidsHit: 0);

            return nativeResult;
        }

        debugHook?.OnFrameExit(
            debugFrame!,
            OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_IMPLEMENTED);

        var backendName =
            string.IsNullOrWhiteSpace(
                _nativeCpuBackend.BackendName)
                ? "native-backend"
                : _nativeCpuBackend.BackendName;

        var backendError =
            string.IsNullOrWhiteSpace(
                _nativeCpuBackend.LastError)
                ? "unknown backend error"
                : _nativeCpuBackend.LastError;

        LastNotImplementedInfo =
            new CpuNotImplementedInfo(
                CpuNotImplementedSource.NativeBackend,
                entryPoint,
                null,
                "cpu_engine_native_only",
                backendName,
                backendError);

        LastMilestoneLog = string.Concat(
            LastMilestoneLog,
            Environment.NewLine,
            $"Native backend failed: {backendError}");

        return FailEarly(
            OrbisGen2Result.ORBIS_GEN2_ERROR_NOT_IMPLEMENTED,
            CpuExitReason.NativeBackendUnavailable);
    }

    // ---------------------------------------------------------------------
    // Stack
    // ---------------------------------------------------------------------

    internal ulong TryMapStackRegion()
    {
        for (var attempt = 0;
             attempt < MaxStackSlots;
             attempt++)
        {
            var slot =
                Interlocked.Increment(
                    ref _nextStackSlot);

            if (slot >= MaxStackSlots)
            {
                return 0;
            }

            var candidate =
                StackBaseAddress -
                ((ulong)slot * StackSlotStride);

            if (!IsValidGuestAddress(
                    candidate,
                    StackSize))
            {
                Console.Error.WriteLine(
                    $"[DISPATCHER] Invalid stack address: 0x{candidate:X16}");

                continue;
            }

            try
            {
                _virtualMemory.Map(
                    candidate,
                    StackSize,
                    fileOffset: 0,
                    fileData: ReadOnlySpan<byte>.Empty,
                    ProgramHeaderFlags.Read |
                    ProgramHeaderFlags.Write);

                Console.Error.WriteLine(
                    $"[DISPATCHER] Stack mapped: 0x{candidate:X16}-0x{candidate + StackSize:X16}");

                return candidate;
            }
            catch (InvalidOperationException)
            {
                Console.Error.WriteLine(
                    $"[DISPATCHER] Stack slot occupied: 0x{candidate:X16}");
            }
        }

        return 0;
    }

    // ---------------------------------------------------------------------
    // TLS
    // ---------------------------------------------------------------------

    internal ulong TryMapTlsRegion()
    {
        var prefix =
            GuestTlsTemplate.StartupStaticTlsReservation;

        for (var attempt = 0;
             attempt < MaxTlsSlots;
             attempt++)
        {
            var slot =
                Interlocked.Increment(
                    ref _nextTlsSlot);

            if (slot >= MaxTlsSlots)
            {
                return 0;
            }

            var tlsBase =
                TlsBaseAddress -
                ((ulong)slot * TlsSlotStride);

            var mappedBase =
                tlsBase -
                prefix;

            var mappedSize =
                TlsSize +
                prefix;

            if (!IsValidGuestAddress(
                    mappedBase,
                    mappedSize))
            {
                Console.Error.WriteLine(
                    $"[DISPATCHER] Invalid TLS address: 0x{mappedBase:X16}");

                continue;
            }

            try
            {
                _virtualMemory.Map(
                    mappedBase,
                    mappedSize,
                    fileOffset: 0,
                    fileData: ReadOnlySpan<byte>.Empty,
                    ProgramHeaderFlags.Read |
                    ProgramHeaderFlags.Write);

                Console.Error.WriteLine(
                    $"[DISPATCHER] TLS mapped: base=0x{tlsBase:X16}");

                return tlsBase;
            }
            catch (InvalidOperationException)
            {
                Console.Error.WriteLine(
                    $"[DISPATCHER] TLS slot occupied: 0x{mappedBase:X16}");
            }
        }

        return 0;
    }

    internal static bool InitializeTls(
        CpuContext context,
        ulong tlsBase)
    {
        /*
         * FreeBSD/amd64-style TLS/TCB compatibility layout.
         */
        if (!context.TryWriteUInt64(
                tlsBase - 0xF0,
                0))
        {
            return false;
        }

        if (!context.TryWriteUInt64(
                tlsBase + 0x00,
                tlsBase))
        {
            return false;
        }

        if (!context.TryWriteUInt64(
                tlsBase + 0x10,
                tlsBase))
        {
            return false;
        }

        if (!context.TryWriteUInt64(
                tlsBase + 0x28,
                0xC0DEC0DECAFEBA00UL))
        {
            return false;
        }

        if (!context.TryWriteUInt64(
                tlsBase + 0x60,
                tlsBase))
        {
            return false;
        }

        GuestTlsTemplate.SeedThreadBlock(
            context,
            tlsBase);

        return true;
    }

    // ---------------------------------------------------------------------
    // Guest stack frame
    // ---------------------------------------------------------------------

    internal static bool InitializeGuestFrameChainSentinel(
        CpuContext context)
    {
        var stackTop =
            context[CpuRegister.Rsp] +
            sizeof(ulong);

        var sentinelFrame =
            AlignDown(
                stackTop - 0x20,
                16);

        var seedRsp =
            sentinelFrame -
            sizeof(ulong);

        if (!context.TryWriteUInt64(
                sentinelFrame,
                0))
        {
            return false;
        }

        if (!context.TryWriteUInt64(
                sentinelFrame + sizeof(ulong),
                0))
        {
            return false;
        }

        if (!context.TryWriteUInt64(
                seedRsp,
                0))
        {
            return false;
        }

        context[CpuRegister.Rbp] =
            sentinelFrame;

        context[CpuRegister.Rsp] =
            seedRsp;

        return true;
    }

    // ---------------------------------------------------------------------
    // Process entry
    // ---------------------------------------------------------------------

    private static bool InitializeProcessEntryFrame(
        CpuContext context,
        string processImageName,
        ulong programExitHandlerAddress)
    {
        var imageName =
            string.IsNullOrWhiteSpace(
                processImageName)
                ? "eboot.bin"
                : processImageName;

        var arguments =
            new List<string>(3)
            {
                imageName
            };

        var configuredArguments =
            Environment.GetEnvironmentVariable(
                "SHARPEMU_GUEST_ARGS");

        if (!string.IsNullOrWhiteSpace(
                configuredArguments))
        {
            var extra =
                configuredArguments.Split(
                    (char[]?)null,
                    StringSplitOptions.RemoveEmptyEntries |
                    StringSplitOptions.TrimEntries);

            arguments.AddRange(
                extra.Take(2));
        }

        var cursor =
            context[CpuRegister.Rsp];

        var addresses =
            new ulong[arguments.Count];

        for (var index = arguments.Count - 1;
             index >= 0;
             index--)
        {
            var bytes =
                Encoding.UTF8.GetBytes(
                    arguments[index] + '\0');

            cursor =
                AlignDown(
                    cursor - (ulong)bytes.Length,
                    16);

            if (!context.Memory.TryWrite(
                    cursor,
                    bytes))
            {
                return false;
            }

            addresses[index] =
                cursor;
        }

        const ulong EntryParamsSize = 0x20;

        var entryParams =
            AlignDown(
                cursor - EntryParamsSize,
                16);

        if (!TryWriteUInt32(
                context,
                entryParams + 0x00,
                (uint)arguments.Count))
        {
            return false;
        }

        if (!TryWriteUInt32(
                context,
                entryParams + 0x04,
                0))
        {
            return false;
        }

        if (!context.TryWriteUInt64(
                entryParams + 0x08,
                addresses[0]))
        {
            return false;
        }

        if (!context.TryWriteUInt64(
                entryParams + 0x10,
                addresses.Length > 1
                    ? addresses[1]
                    : 0))
        {
            return false;
        }

        if (!context.TryWriteUInt64(
                entryParams + 0x18,
                addresses.Length > 2
                    ? addresses[2]
                    : 0))
        {
            return false;
        }

        var entryRsp =
            entryParams -
            sizeof(ulong);

        if (!context.TryWriteUInt64(
                entryRsp,
                0))
        {
            return false;
        }

        context[CpuRegister.Rsp] =
            entryRsp;

        /*
         * SysV-style argument registers.
         */
        context[CpuRegister.Rdi] =
            entryParams;

        context[CpuRegister.Rsi] =
            programExitHandlerAddress;

        context[CpuRegister.Rdx] = 0;
        context[CpuRegister.Rcx] = 0;
        context[CpuRegister.R8] = 0;
        context[CpuRegister.R9] = 0;

        Console.Error.WriteLine(
            $"[DISPATCHER] Entry params: argc={arguments.Count}");

        return true;
    }

    private static bool InitializeModuleInitializerFrame(
        CpuContext context)
    {
        context[CpuRegister.Rdi] = 0;
        context[CpuRegister.Rsi] = 0;
        context[CpuRegister.Rdx] = 0;
        context[CpuRegister.Rcx] = 0;
        context[CpuRegister.R8] = 0;
        context[CpuRegister.R9] = 0;

        return true;
    }

    // ---------------------------------------------------------------------
    // Bootstrap
    // ---------------------------------------------------------------------

    private bool ShouldInjectBootstrapPayload(
        ulong entryPoint)
    {
        Span<byte> probe =
            stackalloc byte[16];

        if (!_virtualMemory.TryRead(
                entryPoint,
                probe))
        {
            return false;
        }

        if (probe.Length <
            BootstrapStartSignature.Length)
        {
            return false;
        }

        for (var i = 0;
             i < BootstrapStartSignature.Length;
             i++)
        {
            if (probe[i] !=
                BootstrapStartSignature[i])
            {
                return false;
            }
        }

        return true;
    }

    private bool TryInstallBootstrapPayload(
        CpuContext context,
        IDictionary<ulong, string> importStubs)
    {
        var stubAddress =
            TryMapBootstrapStubRegion();

        if (stubAddress == 0)
        {
            return false;
        }

        var payloadAddress =
            TryMapBootstrapPayloadRegion();

        if (payloadAddress == 0)
        {
            return false;
        }

        var statusAddress =
            payloadAddress +
            BootstrapStatusOffset;

        if (!TryWriteUInt64(
                payloadAddress,
                stubAddress))
        {
            return false;
        }

        if (!TryWriteUInt64(
                payloadAddress + 0x08,
                statusAddress))
        {
            return false;
        }

        if (!TryWriteUInt64(
                payloadAddress + 0x10,
                statusAddress))
        {
            return false;
        }

        if (!TryWriteUInt64(
                payloadAddress + 0x18,
                statusAddress))
        {
            return false;
        }

        if (!TryWriteUInt64(
                payloadAddress + 0x20,
                statusAddress))
        {
            return false;
        }

        if (!TryWriteUInt64(
                payloadAddress +
                BootstrapPayloadResultOffset,
                statusAddress))
        {
            return false;
        }

        if (!TryWriteUInt32(
                statusAddress,
                0))
        {
            return false;
        }

        importStubs[stubAddress] =
            RuntimeStubNids.BootstrapBridge;

        importStubs[stubAddress + 0x0A] =
            RuntimeStubNids.BootstrapBridge;

        context[CpuRegister.Rdi] =
            payloadAddress;

        Console.Error.WriteLine(
            $"[DISPATCHER] Bootstrap installed: stub=0x{stubAddress:X16} payload=0x{payloadAddress:X16}");

        return true;
    }

    private ulong TryMapBootstrapStubRegion()
    {
        var stubData =
            new byte[(int)StubRegionSize];

        /*
         * INT3 + RET.
         *
         * The interpreter must implement these x86-64 instructions.
         */
        stubData[0] = 0xCC;
        stubData[1] = 0xC3;

        return TryMapExecutableRegion(
            BootstrapStubBaseAddress,
            stubData);
    }

    private ulong TryMapBootstrapPayloadRegion()
    {
        return TryMapWritableRegion(
            BootstrapPayloadBaseAddress);
    }

    private ulong TryMapDynlibFallbackStubRegion()
    {
        var stubData =
            new byte[(int)StubRegionSize];

        /*
         * xor eax,eax
         * ret
         */
        stubData[0] = 0x31;
        stubData[1] = 0xC0;
        stubData[2] = 0xC3;

        return TryMapExecutableRegion(
            DynlibFallbackStubBaseAddress,
            stubData);
    }

    private ulong TryMapReturnToHostStubRegion()
    {
        var stubData =
            new byte[(int)StubRegionSize];

        /*
         * HLT + INT3.
         *
         * X64InterpreterBackend must trap these as a controlled
         * return-to-host condition instead of treating them as
         * an Android native instruction.
         */
        stubData[0] = 0xF4;
        stubData[1] = 0xCC;

        return TryMapExecutableRegion(
            ReturnToHostStubBaseAddress,
            stubData);
    }

    // ---------------------------------------------------------------------
    // Generic mapping helpers
    // ---------------------------------------------------------------------

    private ulong TryMapExecutableRegion(
        ulong baseAddress,
        byte[] data)
    {
        for (var i = 0;
             i < MaxStubSlots;
             i++)
        {
            var candidate =
                baseAddress -
                ((ulong)i * StubStride);

            if (!IsValidGuestAddress(
                    candidate,
                    StubRegionSize))
            {
                continue;
            }

            try
            {
                _virtualMemory.Map(
                    candidate,
                    StubRegionSize,
                    fileOffset: 0,
                    data,
                    ProgramHeaderFlags.Read |
                    ProgramHeaderFlags.Execute);

                return candidate;
            }
            catch (InvalidOperationException)
            {
                continue;
            }
        }

        return 0;
    }

    private ulong TryMapWritableRegion(
        ulong baseAddress)
    {
        for (var i = 0;
             i < MaxStubSlots;
             i++)
        {
            var candidate =
                baseAddress -
                ((ulong)i * StubStride);

            if (!IsValidGuestAddress(
                    candidate,
                    StubRegionSize))
            {
                continue;
            }

            try
            {
                _virtualMemory.Map(
                    candidate,
                    StubRegionSize,
                    fileOffset: 0,
                    fileData: ReadOnlySpan<byte>.Empty,
                    ProgramHeaderFlags.Read |
                    ProgramHeaderFlags.Write);

                return candidate;
            }
            catch (InvalidOperationException)
            {
                continue;
            }
        }

        return 0;
    }

    // ---------------------------------------------------------------------
    // Diagnostics
    // ---------------------------------------------------------------------

    private void ResetDiagnostics()
    {
        LastTrapInfo = null;
        LastMemoryFaultInfo = null;
        LastControlTransferInfo = null;
        LastNotImplementedInfo = null;
        LastImportResolutionTrace = null;
        LastBasicBlockTrace = null;
        LastMilestoneLog = null;
        LastRecentInstructionWindow = null;
        LastRecentControlTransferTrace = null;
        LastSessionSummary = default;
    }

    private static string BuildEntryFrameDiagnostic(
        ulong entryPoint,
        CpuContext context,
        bool sentinelEnabled,
        ulong sentinelValue,
        bool entryParamsConfigured)
    {
        var rsp =
            context[CpuRegister.Rsp];

        var stackValue =
            context.TryReadUInt64(
                rsp,
                out var value)
                ? $"0x{value:X16}"
                : "??";

        return
            $"EntryFrame: " +
            $"entry_rip=0x{entryPoint:X16} " +
            $"initial_rsp=0x{rsp:X16} " +
            $"[rsp]={stackValue} " +
            $"sentinel_enabled={sentinelEnabled} " +
            $"sentinel_value=0x{sentinelValue:X16} " +
            $"entry_params_configured={entryParamsConfigured} " +
            $"android={IsAndroid}";
    }

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    private static ulong AlignDown(
        ulong value,
        ulong alignment)
    {
        if (alignment == 0 ||
            (alignment &
             (alignment - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(alignment));
        }

        return value &
               ~(alignment - 1);
    }

    private static bool TryWriteUInt32(
        CpuContext context,
        ulong address,
        uint value)
    {
        Span<byte> buffer =
            stackalloc byte[sizeof(uint)];

        BinaryPrimitives.WriteUInt32LittleEndian(
            buffer,
            value);

        return context.Memory.TryWrite(
            address,
            buffer);
    }

    private static bool TryWriteUInt64(
        CpuContext context,
        ulong address,
        ulong value)
    {
        Span<byte> buffer =
            stackalloc byte[sizeof(ulong)];

        BinaryPrimitives.WriteUInt64LittleEndian(
            buffer,
            value);

        return context.Memory.TryWrite(
            address,
            buffer);
    }

    private bool TryWriteUInt64(
        ulong address,
        ulong value)
    {
        Span<byte> buffer =
            stackalloc byte[sizeof(ulong)];

        BinaryPrimitives.WriteUInt64LittleEndian(
            buffer,
            value);

        return _virtualMemory.TryWrite(
            address,
            buffer);
    }

    private bool TryWriteUInt32(
        ulong address,
        uint value)
    {
        Span<byte> buffer =
            stackalloc byte[sizeof(uint)];

        BinaryPrimitives.WriteUInt32LittleEndian(
            buffer,
            value);

        return _virtualMemory.TryWrite(
            address,
            buffer);
    }

    private static bool IsValidGuestAddress(
        ulong address,
        ulong size)
    {
        if (size == 0)
        {
            return false;
        }

        /*
         * Conservative 39-bit guest VA limit.
         *
         * 2^39 = 0x8000000000
         */
        const ulong Max39BitAddress =
            0x0000_8000_0000_0000UL;

        if (address >= Max39BitAddress)
        {
            return false;
        }

        if (size >
            Max39BitAddress - address)
        {
            return false;
        }

        return true;
    }

    private static CpuExecutionOptions ForceInterpreter(
        CpuExecutionOptions options)
    {
        /*
         * CpuExecutionOptions is a struct. We intentionally copy it instead
         * of introducing a new API surface.
         *
         * The project already uses CpuEngine to select the backend.
         */
        return new CpuExecutionOptions
        {
            CpuEngine = CpuExecutionEngine.Interpreter,
            InterpreterTrace = options.InterpreterTrace,
            DebugHook = options.DebugHook
        };
    }

    private OrbisGen2Result FailEarly(
        OrbisGen2Result result,
        CpuExitReason reason =
            CpuExitReason.UnhandledException)
    {
        LastSessionSummary =
            new CpuSessionSummary(
                result,
                reason,
                exitCode: null,
                lastGuestRip:
                    LastEntryPoint ?? 0,
                lastStubRip: 0,
                totalInstructions: 0,
                importsHit: 0,
                uniqueNidsHit: 0);

        return result;
    }

    // ---------------------------------------------------------------------
    // Lifetime
    // ---------------------------------------------------------------------

    public void Dispose()
    {
        if (_nativeCpuBackend is IDisposable disposable)
        {
            disposable.Dispose();
        }

        NativeSessionLeaked =
            _nativeCpuBackend is
            DirectExecutionBackend
            {
                GuestSessionLeaked: true
            };

        _nativeCpuBackend = null;
    }
}
