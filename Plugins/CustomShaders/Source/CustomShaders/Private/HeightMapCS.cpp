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
        SHADER_PARAMETER(int32, NumObjects)
        SHADER_PARAMETER_RDG_BUFFER_SRV(StructuredBuffer<float>, WaveHeights)
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
    FUploadParams* UploadParams = GraphBuilder.AllocParameters<FUploadParams>(); UploadParams->Target = Buffer; GraphBuilder.AddPass( RDG_EVENT_NAME("Upload_%s", Name), FUploadParams::FTypeInfo::GetStructMetadata(), UploadParams, ERDGPassFlags::Copy, [Buffer, InitialData, SizeInBytes](FRHICommandList& RHICmdList) { void* Dest = RHICmdList.LockBuffer(Buffer->GetRHI(), 0, SizeInBytes, RLM_WriteOnly); FMemory::Memcpy(Dest, InitialData, SizeInBytes); RHICmdList.UnlockBuffer(Buffer->GetRHI()); } );
    return Buffer;
}

bool GenerateHeightMapGPU(const FHeightMapGenParams& P,
                          const TArray<float>& WaveHeights,
                          const TArray<FHeightMapObject>& Objects,
                          TArray<float>& OutState)
{

    if (P.GridSize <= 0 || P.StateW <= 0 || P.StateH <= 0)
    {
        UE_LOG(LogTemp, Error, TEXT("GenerateHeightMapGPU: Invalid parameters"));
        return false;
    }
    if (WaveHeights.Num() < P.GridSize * P.GridSize)
    {
        UE_LOG(LogTemp, Error, TEXT("GenerateHeightMapGPU: Insufficient wave data. Need %d, got %d"), P.GridSize * P.GridSize, WaveHeights.Num());
        return false;
    }

    const int32 OutCount = P.StateW * P.StateH;
    OutState.SetNumUninitialized(OutCount);

    // Initialize with a known pattern for debugging
    for (int32 i = 0; i < OutState.Num(); ++i)
    {
        OutState[i] = -999.0f; // Known pattern to verify if GPU overwrites it
    }

    // Use a shared pointer to pass data back from render thread
    TSharedPtr<TArray<float>, ESPMode::ThreadSafe> SharedResult = MakeShared<TArray<float>, ESPMode::ThreadSafe>();
    SharedResult->SetNumZeroed(OutCount);

    ENQUEUE_RENDER_COMMAND(HeightMapDispatch)([P, WaveHeightsCopy = WaveHeights, ObjectsCopy = Objects, SharedResult](FRHICommandListImmediate& RHICmdList)
    {
        FRDGBuilder GraphBuilder(RHICmdList);

        // Create SRV buffers for inputs
        const int32 WaveCount = P.GridSize * P.GridSize;
        FRDGBufferRef WaveBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("WaveHeights"), sizeof(float), WaveCount, WaveHeightsCopy.GetData());
        FRDGBufferSRVRef WaveSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(WaveBuf));

        const int32 NumObjs = ObjectsCopy.Num();
        TArray<FVector3f> Centers; Centers.Reserve(NumObjs);
        TArray<FVector3f> Radii;   Radii.Reserve(NumObjs);
        for (const FHeightMapObject& O : ObjectsCopy)
        {
            Centers.Add(FVector3f((float)O.CenterLocal.X, (float)O.CenterLocal.Y, (float)O.CenterLocal.Z));
            Radii.Add(FVector3f((float)O.Radii.X, (float)O.Radii.Y, (float)O.Radii.Z));
        }
        // Always create buffers, even if empty, to avoid null binding
        const int32 ActualNumObjs = FMath::Max(NumObjs, 1);
        if (NumObjs == 0)
        {
            Centers.Add(FVector3f::ZeroVector);
            Radii.Add(FVector3f::ZeroVector);
        }

        FRDGBufferRef CentersBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("ObjCenters"), sizeof(FVector3f), ActualNumObjs, Centers.GetData());
        FRDGBufferSRVRef CentersSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(CentersBuf));
        FRDGBufferRef RadiiBuf = CreateStructuredBufferWithData(GraphBuilder, TEXT("ObjRadii"), sizeof(FVector3f), ActualNumObjs, Radii.GetData());
        FRDGBufferSRVRef RadiiSRV = GraphBuilder.CreateSRV(FRDGBufferSRVDesc(RadiiBuf));

        // Output buffer
        const int32 OutCount = P.StateW * P.StateH;
        FRDGBufferRef OutBuf = GraphBuilder.CreateBuffer(FRDGBufferDesc::CreateStructuredDesc(sizeof(float), OutCount), TEXT("OutHeightMap"));
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
        Params->NumObjects = NumObjs;
        Params->WaveHeights = WaveSRV;
        Params->ObjCenters = CentersSRV;
        Params->ObjRadii = RadiiSRV;
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
        AddEnqueueCopyPass(GraphBuilder, &Readback, OutBuf, OutCount * sizeof(float));
        GraphBuilder.Execute();

        void* DataPtr = Readback.Lock(OutCount * sizeof(float));
        if (DataPtr)
        {
            FMemory::Memcpy(SharedResult->GetData(), DataPtr, OutCount * sizeof(float));
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("GenerateHeightMapGPU: Failed to lock readback buffer"));
            FMemory::Memzero(SharedResult->GetData(), OutCount * sizeof(float));
        }
        Readback.Unlock();
    });

    // Flush to ensure completion before returning
    FlushRenderingCommands();

    // Copy result back to OutState
    OutState = *SharedResult;

    return true;
}






