#include "HeightMapGenerator.h"

#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "RenderGraphUtils.h"
#include "RHIStaticStates.h"
#include "RHI.h"
#include "RHIGPUReadback.h"

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

    const int32 OutCount = P.StateW * P.StateH;
    OutState.SetNumUninitialized(OutCount);
    for (int32 i = 0; i < OutCount; ++i) { OutState[i] = -999.0f; }

    TSharedPtr<TArray<float>, ESPMode::ThreadSafe> SharedResult = MakeShared<TArray<float>, ESPMode::ThreadSafe>();
    SharedResult->SetNumZeroed(OutCount);

    ENQUEUE_RENDER_COMMAND(HeightMapDispatch)([P, ColumnCentersCopy = ColumnCentersIn, ColumnRadiiCopy = ColumnRadiiIn, ObjectsCopy = Objects, SharedResult](FRHICommandListImmediate& RHICmdList)
    {
        FRDGBuilder GraphBuilder(RHICmdList);

        // Pack columns to FVector3f
        const int32 ColCount = P.GridSize * P.GridSize;
        TArray<FVector3f> Centers3f; Centers3f.SetNumUninitialized(ColCount);
        TArray<FVector3f> Radii3f;   Radii3f.SetNumUninitialized(ColCount);
        for (int32 i = 0; i < ColCount; ++i)
        {
            const FVector& C = ColumnCentersCopy[i];
            Centers3f[i] = FVector3f((float)C.X, (float)C.Y, (float)C.Z);
            const FVector& R = ColumnRadiiCopy[i];
            Radii3f[i] = FVector3f((float)R.X, (float)R.Y, (float)R.Z);
        }
        FRDGBufferRef ColCentersBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("ColumnCenters"), sizeof(FVector3f), ColCount, Centers3f.GetData());
        FRDGBufferSRVRef ColCentersSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ColCentersBuf));
        FRDGBufferRef ColRadiiBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("ColumnRadiiArray"), sizeof(FVector3f), ColCount, Radii3f.GetData());
        FRDGBufferSRVRef ColRadiiSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ColRadiiBuf));

        // Pack objects
        const int32 NumObjs = ObjectsCopy.Num();
        TArray<FVector3f> ObjCenters3f; ObjCenters3f.Reserve(NumObjs > 0 ? NumObjs : 1);
        TArray<FVector3f> ObjRadii3f;   ObjRadii3f.Reserve(NumObjs > 0 ? NumObjs : 1);
        for (const FHeightMapObject& O : ObjectsCopy)
        {
            ObjCenters3f.Add(FVector3f((float)O.CenterLocal.X, (float)O.CenterLocal.Y, (float)O.CenterLocal.Z));
            ObjRadii3f.Add(FVector3f((float)O.Radii.X, (float)O.Radii.Y, (float)O.Radii.Z));
        }
        if (NumObjs == 0)
        {
            ObjCenters3f.Add(FVector3f::ZeroVector);
            ObjRadii3f.Add(FVector3f::ZeroVector);
        }
        FRDGBufferRef ObjCentersBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("ObjCenters"), sizeof(FVector3f), ObjCenters3f.Num(), ObjCenters3f.GetData());
        FRDGBufferSRVRef ObjCentersSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ObjCentersBuf));
        FRDGBufferRef ObjRadiiBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("ObjRadii"), sizeof(FVector3f), ObjRadii3f.Num(), ObjRadii3f.GetData());
        FRDGBufferSRVRef ObjRadiiSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(ObjRadiiBuf));

        // Output buffer
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
        Params->NumObjects = NumObjs;
        Params->ColumnCenters = ColCentersSRV;
        Params->ColumnRadiiArray = ColRadiiSRV;
        Params->ObjCenters = ObjCentersSRV;
        Params->ObjRadii = ObjRadiiSRV;
        Params->OutHeightMap = OutUAV;

        TShaderMapRef<FHeightMapCS> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
        if (!CS.IsValid())
        {
            UE_LOG(LogTemp, Error, TEXT("GenerateHeightMapGPU: Shader not found or invalid"));
            return;
        }

        const FIntVector GroupCounts(FMath::DivideAndRoundUp(P.StateW, 16), FMath::DivideAndRoundUp(P.StateH, 16), 1);
        FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("HeightMapCS"), ERDGPassFlags::Compute, CS, Params, GroupCounts);

        // Readback
        static FLazyName ReadbackName(TEXT("HeightMapReadback"));
        FRHIGPUBufferReadback Readback(ReadbackName);
        AddEnqueueCopyPass(GraphBuilder, &Readback, OutBuf, P.StateW * P.StateH * sizeof(float));
        GraphBuilder.Execute();

        void* DataPtr = Readback.Lock(P.StateW * P.StateH * sizeof(float));
        if (DataPtr)
        {
            FMemory::Memcpy(SharedResult->GetData(), DataPtr, P.StateW * P.StateH * sizeof(float));
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("GenerateHeightMapGPU: Failed to lock readback buffer"));
            FMemory::Memzero(SharedResult->GetData(), P.StateW * P.StateH * sizeof(float));
        }
        Readback.Unlock();
    });

    FlushRenderingCommands();
    OutState = *SharedResult;
    return true;
}
