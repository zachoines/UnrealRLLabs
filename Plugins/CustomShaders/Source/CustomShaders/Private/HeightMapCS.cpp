#include "HeightMapGenerator.h"

#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "RenderGraphUtils.h"
#include "RHIStaticStates.h"
#include "RHI.h"
#include "RHIGPUReadback.h"
#include "HAL/Event.h"
#include "HAL/PlatformProcess.h"

class FHeightMapCS : public FGlobalShader
{
public:
    DECLARE_GLOBAL_SHADER(FHeightMapCS);
    SHADER_USE_PARAMETER_STRUCT(FHeightMapCS, FGlobalShader);

    BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
        SHADER_PARAMETER(FVector2f, PlatformSize)
        SHADER_PARAMETER(int32, GridSize)
        SHADER_PARAMETER(int32, StateW)
        SHADER_PARAMETER(int32, StateH)
        SHADER_PARAMETER(float, CellSize)
        SHADER_PARAMETER(float, MinZ)
        SHADER_PARAMETER(float, MaxZ)
        SHADER_PARAMETER(FVector3f, ColumnRadii)
        SHADER_PARAMETER(float, ColZBias)
        SHADER_PARAMETER(float, ObjZBias)
        SHADER_PARAMETER(int32, NumObjects)
        SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FVector3f>, ColumnCenters)
        SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FVector3f>, ColumnRadiiArray)
        SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FVector3f>, ObjCenters)
        SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<FVector3f>, ObjRadii)
        SHADER_PARAMETER_RDG_BUFFER_UAV(RWStructuredBuffer<float>, OutHeightMap)
    END_SHADER_PARAMETER_STRUCT()

    static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters) { return true; }
};

IMPLEMENT_GLOBAL_SHADER(FHeightMapCS, "/Plugins/CustomShaders/HeightMapCS.usf", "MainCS", SF_Compute);

BEGIN_SHADER_PARAMETER_STRUCT(FUploadParams, )
    RDG_BUFFER_ACCESS(Target, ERHIAccess::CopyDest)
END_SHADER_PARAMETER_STRUCT()

static FRDGBufferRef CreateStructuredBufferWithData(FRDGBuilder& GraphBuilder, const TCHAR* Name, uint32 Stride, uint32 NumElements, const void* InitialData)
{
    check(NumElements > 0);
    const uint64 SizeInBytes = (uint64)Stride * (uint64)NumElements;
    FRDGBufferRef Buffer = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(Stride, NumElements), Name);
    FUploadParams* UploadParams = GraphBuilder.AllocParameters<FUploadParams>();
    UploadParams->Target = Buffer;
    GraphBuilder.AddPass(
        RDG_EVENT_NAME("Upload_%s", Name),
        FUploadParams::FTypeInfo::GetStructMetadata(),
        UploadParams,
        ERDGPassFlags::Copy,
        [Buffer, InitialData, SizeInBytes](FRHICommandList& RHICmdList)
        {
            void* Dest = RHICmdList.LockBuffer(Buffer->GetRHI(), 0, SizeInBytes, RLM_WriteOnly);
            FMemory::Memcpy(Dest, InitialData, SizeInBytes);
            RHICmdList.UnlockBuffer(Buffer->GetRHI());
        }
    );
    return Buffer;
}

bool DispatchHeightMapGPU(const FHeightMapGenParams& P,
                          const TArray<FVector3f>& ColumnCenters,
                          const TArray<FVector3f>& ColumnRadiiIn,
                          const TArray<FVector3f>& ObjectCenters,
                          const TArray<FVector3f>& ObjectRadii,
                          FHeightMapGPUDispatchHandle& InOutHandle)
{
    if (InOutHandle.IsActive())
    {
        UE_LOG(LogTemp, Warning, TEXT("DispatchHeightMapGPU: Previous readback still in flight"));
        return false;
    }

    if (P.GridSize <= 0 || P.StateW <= 0 || P.StateH <= 0)
    {
        UE_LOG(LogTemp, Error, TEXT("DispatchHeightMapGPU: Invalid dimensions (Grid=%d, StateW=%d, StateH=%d)"), P.GridSize, P.StateW, P.StateH);
        return false;
    }

    const int32 ColCount = P.GridSize * P.GridSize;
    if (ColumnCenters.Num() != ColCount || ColumnRadiiIn.Num() != ColCount)
    {
        UE_LOG(LogTemp, Error, TEXT("DispatchHeightMapGPU: Expected %d column entries (got %d centers / %d radii)"), ColCount, ColumnCenters.Num(), ColumnRadiiIn.Num());
        return false;
    }

    if (ObjectCenters.Num() != ObjectRadii.Num())
    {
        UE_LOG(LogTemp, Error, TEXT("DispatchHeightMapGPU: Object centers/radii mismatch (%d vs %d)"), ObjectCenters.Num(), ObjectRadii.Num());
        return false;
    }

    const int32 OutCount = P.StateW * P.StateH;
    if (OutCount <= 0)
    {
        UE_LOG(LogTemp, Error, TEXT("DispatchHeightMapGPU: Output element count must be positive"));
        return false;
    }

    if (!InOutHandle.Readback.IsValid())
    {
        static FLazyName ReadbackName(TEXT("HeightMapReadback"));
        InOutHandle.Readback = MakeShared<FRHIGPUBufferReadback, ESPMode::ThreadSafe>(ReadbackName);
    }

    const TSharedPtr<FRHIGPUBufferReadback, ESPMode::ThreadSafe> ReadbackPtr = InOutHandle.Readback;
    static const FVector3f ZeroVector = FVector3f::ZeroVector;
    const FVector3f* ObjCentersPtr = ObjectCenters.Num() > 0 ? ObjectCenters.GetData() : &ZeroVector;
    const FVector3f* ObjRadiiPtr = ObjectRadii.Num() > 0 ? ObjectRadii.GetData() : &ZeroVector;
    const int32 ObjBufferCount = FMath::Max(ObjectCenters.Num(), 1);

    const FVector3f* ColCentersPtr = ColumnCenters.GetData();
    const FVector3f* ColRadiiPtr = ColumnRadiiIn.GetData();

    const uint64 BytesToCopy = (uint64)OutCount * sizeof(float);

    InOutHandle.ExpectedBytes = BytesToCopy;
    InOutHandle.ElementCount = OutCount;

    ENQUEUE_RENDER_COMMAND(HeightMapDispatch)([P, ColCentersPtr, ColRadiiPtr, ObjCentersPtr, ObjRadiiPtr, ColCount, ObjBufferCount, ReadbackPtr, BytesToCopy](FRHICommandListImmediate& RHICmdList)
    {
        FRDGBuilder GraphBuilder(RHICmdList);

        FRDGBufferRef ColCentersBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("ColumnCenters"), sizeof(FVector3f), ColCount, ColCentersPtr);
        FRDGBufferSRVRef ColCentersSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ColCentersBuf));

        FRDGBufferRef ColRadiiBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("ColumnRadiiArray"), sizeof(FVector3f), ColCount, ColRadiiPtr);
        FRDGBufferSRVRef ColRadiiSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ColRadiiBuf));

        FRDGBufferRef ObjCentersBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("ObjCenters"), sizeof(FVector3f), ObjBufferCount, ObjCentersPtr);
        FRDGBufferSRVRef ObjCentersSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ObjCentersBuf));

        FRDGBufferRef ObjRadiiBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("ObjRadii"), sizeof(FVector3f), ObjBufferCount, ObjRadiiPtr);
        FRDGBufferSRVRef ObjRadiiSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ObjRadiiBuf));

        FRDGBufferRef OutBuf = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(float), P.StateW * P.StateH), TEXT("OutHeightMap"));
        FRDGBufferUAVRef OutUAV = GraphBuilder.CreateUAV(FRDGBufferUAVDesc(OutBuf));

        FHeightMapCS::FParameters* Params = GraphBuilder.AllocParameters<FHeightMapCS::FParameters>();
        Params->PlatformSize = FVector2f((float)P.PlatformSize.X, (float)P.PlatformSize.Y);
        Params->GridSize = P.GridSize;
        Params->StateW = P.StateW;
        Params->StateH = P.StateH;
        Params->CellSize = P.CellSize;
        Params->MinZ = P.MinZ;
        Params->MaxZ = P.MaxZ;
        Params->ColumnRadii = FVector3f((float)P.ColumnRadii.X, (float)P.ColumnRadii.Y, (float)P.ColumnRadii.Z);
        Params->ColZBias = P.ColZBias;
        Params->ObjZBias = P.ObjZBias;
        Params->NumObjects = P.NumObjects;
        Params->ColumnCenters = ColCentersSRV;
        Params->ColumnRadiiArray = ColRadiiSRV;
        Params->ObjCenters = ObjCentersSRV;
        Params->ObjRadii = ObjRadiiSRV;
        Params->OutHeightMap = OutUAV;

        TShaderMapRef<FHeightMapCS> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
        if (!CS.IsValid())
        {
            UE_LOG(LogTemp, Error, TEXT("DispatchHeightMapGPU: Shader not found or invalid"));
            return;
        }

        const FIntVector GroupCounts(FMath::DivideAndRoundUp(P.StateW, 16), FMath::DivideAndRoundUp(P.StateH, 16), 1);
        FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("HeightMapCS"), ERDGPassFlags::Compute, CS, Params, GroupCounts);

        AddEnqueueCopyPass(GraphBuilder, ReadbackPtr.Get(), OutBuf, BytesToCopy);
        GraphBuilder.Execute();
    });

    InOutHandle.bInFlight = true;
    return true;
}

static bool ResolveHeightMapGPU_RenderThread(FHeightMapGPUDispatchHandle& Handle, TArray<float>& OutState)
{
    check(IsInRenderingThread());

    if (Handle.ExpectedBytes == 0 || Handle.ElementCount == 0)
    {
        UE_LOG(LogTemp, Warning, TEXT("ResolveHeightMapGPU: Handle has zero-sized readback"));
        Handle.Reset();
        return false;
    }

    void* DataPtr = Handle.Readback->Lock(Handle.ExpectedBytes);
    if (!DataPtr)
    {
        UE_LOG(LogTemp, Error, TEXT("ResolveHeightMapGPU: Failed to lock readback buffer"));
        Handle.Reset();
        return false;
    }

    OutState.SetNum(Handle.ElementCount, EAllowShrinking::No);
    FMemory::Memcpy(OutState.GetData(), DataPtr, Handle.ExpectedBytes);
    Handle.Readback->Unlock();
    Handle.Reset();
    return true;
}

bool ResolveHeightMapGPU(FHeightMapGPUDispatchHandle& Handle, TArray<float>& OutState)
{
    if (!Handle.Readback.IsValid() || !Handle.IsActive())
    {
        return false;
    }

    if (!Handle.Readback->IsReady())
    {
        return false;
    }

    if (IsInRenderingThread())
    {
        return ResolveHeightMapGPU_RenderThread(Handle, OutState);
    }

    bool bResolved = false;
    FEvent* CompletionEvent = FPlatformProcess::GetSynchEventFromPool(true);
    ENQUEUE_RENDER_COMMAND(ResolveHeightMapGPUCommand)([HandlePtr = &Handle, OutStatePtr = &OutState, CompletionEvent, &bResolved](FRHICommandListImmediate& RHICmdList)
    {
        bResolved = ResolveHeightMapGPU_RenderThread(*HandlePtr, *OutStatePtr);
        CompletionEvent->Trigger();
    });

    CompletionEvent->Wait();
    FPlatformProcess::ReturnSynchEventToPool(CompletionEvent);
    return bResolved;
}

bool GenerateHeightMapGPU(const FHeightMapGenParams& P,
                          const TArray<FVector>& ColumnCentersIn,
                          const TArray<FVector>& ColumnRadiiIn,
                          const TArray<FHeightMapObject>& Objects,
                          TArray<float>& OutState)
{
    if (P.GridSize <= 0 || P.StateW <= 0 || P.StateH <= 0)
    {
        UE_LOG(LogTemp, Error, TEXT("GenerateHeightMapGPU: Invalid parameters"));
        return false;
    }
    const int32 ColCount = P.GridSize * P.GridSize;
    if (ColumnCentersIn.Num() < ColCount || ColumnRadiiIn.Num() < ColCount)
    {
        UE_LOG(LogTemp, Error, TEXT("GenerateHeightMapGPU: Need per-column centers and radii sized N*N (got %d, %d) for N=%d"), ColumnCentersIn.Num(), ColumnRadiiIn.Num(), P.GridSize);
        return false;
    }

    FHeightMapGenParams ParamsCopy = P;

    TArray<FVector3f> ColumnCenters3f;
    ColumnCenters3f.Reserve(ColCount);
    for (int32 i = 0; i < ColCount; ++i)
    {
        const FVector& C = ColumnCentersIn[i];
        ColumnCenters3f.Add(FVector3f((float)C.X, (float)C.Y, (float)C.Z));
    }

    TArray<FVector3f> ColumnRadii3f;
    ColumnRadii3f.Reserve(ColCount);
    for (int32 i = 0; i < ColCount; ++i)
    {
        const FVector& R = ColumnRadiiIn[i];
        ColumnRadii3f.Add(FVector3f((float)R.X, (float)R.Y, (float)R.Z));
    }

    TArray<FVector3f> ObjCenters3f;
    TArray<FVector3f> ObjRadii3f;
    ObjCenters3f.Reserve(Objects.Num());
    ObjRadii3f.Reserve(Objects.Num());
    for (const FHeightMapObject& O : Objects)
    {
        ObjCenters3f.Add(FVector3f((float)O.CenterLocal.X, (float)O.CenterLocal.Y, (float)O.CenterLocal.Z));
        ObjRadii3f.Add(FVector3f((float)O.Radii.X, (float)O.Radii.Y, (float)O.Radii.Z));
    }

    ParamsCopy.NumObjects = ObjCenters3f.Num();
    if (ParamsCopy.NumObjects == 0)
    {
        ObjCenters3f.Add(FVector3f::ZeroVector);
        ObjRadii3f.Add(FVector3f::ZeroVector);
    }

    FHeightMapGPUDispatchHandle Handle;
    if (!DispatchHeightMapGPU(ParamsCopy, ColumnCenters3f, ColumnRadii3f, ObjCenters3f, ObjRadii3f, Handle))
    {
        return false;
    }

    FlushRenderingCommands();

    if (!ResolveHeightMapGPU(Handle, OutState))
    {
        UE_LOG(LogTemp, Error, TEXT("GenerateHeightMapGPU: Failed to resolve readback"));
        return false;
    }

    return true;
}

