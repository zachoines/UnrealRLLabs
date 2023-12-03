#include "RayGenTest.h"

#define NUM_THREADS_PER_GROUP_DIMENSION 8

//class FRayGenTestRGS : public FGlobalShader
//{
//	DECLARE_GLOBAL_SHADER(FRayGenTestRGS)
//	SHADER_USE_ROOT_PARAMETER_STRUCT(FRayGenTestRGS, FGlobalShader)
//	
//	BEGIN_SHADER_PARAMETER_STRUCT(FParameters,)
//		SHADER_PARAMETER_UAV(RWTexture2D<float4>, outTex)
//		SHADER_PARAMETER_SRV(RaytracingAccelerationStructure, TLAS)
//		SHADER_PARAMETER_STRUCT_REF(FViewUniformShaderParameters, ViewUniformBuffer)
//	END_SHADER_PARAMETER_STRUCT()
//
//	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
//	{
//		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
//	}
//
//	static ERayTracingPayloadType GetRayTracingPayloadType(const int32 /*PermutationId*/)
//	{
//		return ERayTracingPayloadType::Minimal;
//	}
//
//	static inline void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
//	{
//		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
//
//		//We're using it here to add some preprocessor defines. That way we don't have to change both C++ and HLSL code when we change the value for NUM_THREADS_PER_GROUP_DIMENSION
//		OutEnvironment.SetDefine(TEXT("THREADGROUPSIZE_X"), NUM_THREADS_PER_GROUP_DIMENSION);
//		OutEnvironment.SetDefine(TEXT("THREADGROUPSIZE_Y"), NUM_THREADS_PER_GROUP_DIMENSION);
//		OutEnvironment.SetDefine(TEXT("THREADGROUPSIZE_Z"), NUM_THREADS_PER_GROUP_DIMENSION);
//	}
//};
//IMPLEMENT_GLOBAL_SHADER(FRayGenTestRGS, "/Plugin/Shaders/MyRayTraceTest.usf", "RayTraceTestRGS", SF_RayGen);
//
//class FRayGenTestCHS : public FGlobalShader
//{
//	DECLARE_GLOBAL_SHADER(FRayGenTestCHS)
//	SHADER_USE_ROOT_PARAMETER_STRUCT(FRayGenTestCHS, FGlobalShader)
//
//	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
//	{
//		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
//	}
//
//	static ERayTracingPayloadType GetRayTracingPayloadType(const int32 PermutationId)
//	{
//		return FRayGenTestRGS::GetRayTracingPayloadType(PermutationId);
//	}
//
//	using FParameters = FEmptyShaderParameters;
//};
//IMPLEMENT_GLOBAL_SHADER(FRayGenTestCHS, "/Plugin/Shaders/MyRayTraceTest.usf", "closestHit=RayTraceTestCHS", SF_RayHitGroup);
//
//class FRayGenTestMS : public FGlobalShader
//{
//	DECLARE_GLOBAL_SHADER(FRayGenTestMS)
//	SHADER_USE_ROOT_PARAMETER_STRUCT(FRayGenTestMS, FGlobalShader)
//
//		static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
//	{
//		return ShouldCompileRayTracingShadersForProject(Parameters.Platform);
//	}
//
//	static ERayTracingPayloadType GetRayTracingPayloadType(const int32 PermutationId)
//	{
//		return FRayGenTestRGS::GetRayTracingPayloadType(PermutationId);
//	}
//
//	using FParameters = FEmptyShaderParameters;
//};
//IMPLEMENT_GLOBAL_SHADER(FRayGenTestMS, "/Plugin/Shaders/MyRayTraceTest.usf", "RayTraceTestMS", SF_RayMiss);
//
FRayGenTest::FRayGenTest()
{
}
//
//void FRayGenTest::BeginRendering()
//{
//	//If the handle is already initialized and valid, no need to do anything
//	if (PostOpaqueRenderDelegate.IsValid())
//	{
//		return;
//	}
//	//Get the Renderer Module and add our entry to the callbacks so it can be executed each frame after the scene rendering is done
//	const FName RendererModuleName("Renderer");
//	IRendererModule* RendererModule = FModuleManager::GetModulePtr<IRendererModule>(RendererModuleName);
//	if (RendererModule)
//	{
//		PostOpaqueRenderDelegate = RendererModule->RegisterPostOpaqueRenderDelegate(FPostOpaqueRenderDelegate::CreateRaw(this, &FRayGenTest::Execute_RenderThread));
//	}
//
//	// create output texture
//	FIntPoint TextureSize = { CachedParams.RenderTarget->SizeX, CachedParams.RenderTarget->SizeY };
//	FRHITextureCreateDesc TextureDesc = FRHITextureCreateDesc::Create2D(TEXT("RaytracingTestOutput"), TextureSize.X, TextureSize.Y, CachedParams.RenderTarget->GetFormat());
//	TextureDesc.AddFlags(TexCreate_ShaderResource | TexCreate_UAV);
//	ShaderOutputTexture = RHICreateTexture(TextureDesc);
//	ShaderOutputTextureUAV = RHICreateUnorderedAccessView(ShaderOutputTexture);
//}
//
////Stop the compute shader execution
//void FRayGenTest::EndRendering()
//{
//	//If the handle is not valid then there's no cleanup to do
//	if (!PostOpaqueRenderDelegate.IsValid())
//	{
//		return;
//	}
//	//Get the Renderer Module and remove our entry from the PostOpaqueRender callback
//	const FName RendererModuleName("Renderer");
//	IRendererModule* RendererModule = FModuleManager::GetModulePtr<IRendererModule>(RendererModuleName);
//	if (RendererModule)
//	{
//		RendererModule->RemovePostOpaqueRenderDelegate(PostOpaqueRenderDelegate);
//	}
//
//	PostOpaqueRenderDelegate.Reset();
//}

void FRayGenTest::UpdateParameters(FRayGenTestParameters& DrawParameters)
{
	CachedParams = DrawParameters;
	bCachedParamsAreValid = true;
}

//void FRayGenTest::Execute_RenderThread(FPostOpaqueRenderParameters& Parameters)
//#if RHI_RAYTRACING
//{
//	FRDGBuilder* GraphBuilder = Parameters.GraphBuilder;
//	FRHICommandListImmediate& RHICmdList = GraphBuilder->RHICmdList;
//	//If there's no cached parameters to use, skip
//	//If no Render Target is supplied in the cachedParams, skip
//	if (!(bCachedParamsAreValid && CachedParams.RenderTarget))
//	{
//		return;
//	}
//
//	//Render Thread Assertion
//	check(IsInRenderingThread());
//
//	// set shader parameters
//	FRayGenTestRGS::FParameters *PassParameters = GraphBuilder->AllocParameters<FRayGenTestRGS::FParameters>();
//	PassParameters->ViewUniformBuffer = Parameters.View->ViewUniformBuffer;
//	// FRHIShaderResourceView
//	// PassParameters->TLAS = CachedParams.Scene->;
//	PassParameters->outTex = ShaderOutputTextureUAV;
//
//	// define render pass needed parameters
//	TShaderMapRef<FRayGenTestRGS> RayGenTestRGS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
//	FIntPoint TextureSize = { CachedParams.RenderTarget->SizeX, CachedParams.RenderTarget->SizeY };
//	FRHIRayTracingScene* RHIScene = CachedParams.Scene->GetRHIRayTracingScene();
//
//	// add the ray trace dispatch pass
//	GraphBuilder->AddPass(
//		RDG_EVENT_NAME("RayGenTest"),
//		PassParameters,
//		ERDGPassFlags::Compute,
//		[PassParameters, RayGenTestRGS, TextureSize, RHIScene](FRHIRayTracingCommandList& RHICmdList)
//		{
//			FRayTracingShaderBindingsWriter GlobalResources;
//			SetShaderParameters(GlobalResources, RayGenTestRGS, *PassParameters);
//
//			FRayTracingPipelineStateInitializer PSOInitializer;
//			PSOInitializer.MaxPayloadSizeInBytes = GetRayTracingPayloadTypeMaxSize(FRayGenTestRGS::GetRayTracingPayloadType(0));;
//			PSOInitializer.bAllowHitGroupIndexing = false;
//
//			// Set RayGen shader
//			TArray<FRHIRayTracingShader*> RayGenShaderTable;
//			RayGenShaderTable.Add(GetGlobalShaderMap(GMaxRHIFeatureLevel)->GetShader<FRayGenTestRGS>().GetRayTracingShader());
//			PSOInitializer.SetRayGenShaderTable(RayGenShaderTable);
//
//			// Set ClosestHit shader
//			TArray<FRHIRayTracingShader*> RayHitShaderTable;
//			RayHitShaderTable.Add(GetGlobalShaderMap(GMaxRHIFeatureLevel)->GetShader<FRayGenTestCHS>().GetRayTracingShader());
//			PSOInitializer.SetHitGroupTable(RayHitShaderTable);
//			
//			// Set Miss shader
//			TArray<FRHIRayTracingShader*> RayMissShaderTable;
//			RayMissShaderTable.Add(GetGlobalShaderMap(GMaxRHIFeatureLevel)->GetShader<FRayGenTestMS>().GetRayTracingShader());
//			PSOInitializer.SetMissShaderTable(RayMissShaderTable);
//
//			// dispatch ray trace shader
//			FRayTracingPipelineState* PipeLine = PipelineStateCache::GetAndOrCreateRayTracingPipelineState(RHICmdList, PSOInitializer);
//			RHICmdList.SetRayTracingMissShader(RHIScene, 0, PipeLine, 0 /* ShaderIndexInPipeline */, 0, nullptr, 0);
//			RHICmdList.RayTraceDispatch(PipeLine, RayGenTestRGS.GetRayTracingShader(), RHIScene, GlobalResources, TextureSize.X, TextureSize.Y);
//		}
//	);
//
//	// Copy textures from the shader output to our render target
//	// this is done as a render pass with the graph builder
//	FTexture2DRHIRef OriginalRT = CachedParams.RenderTarget->GetRenderTargetResource()->GetTexture2DRHI();
//	FRDGTexture* OutputRDGTexture = GraphBuilder->RegisterExternalTexture(CreateRenderTarget(ShaderOutputTexture, TEXT("RaytracingTestOutputRT")));
//	FRDGTexture* CopyToRDGTexture = GraphBuilder->RegisterExternalTexture(CreateRenderTarget(OriginalRT, TEXT("RaytracingTestCopyToRT")));
//	FRHICopyTextureInfo CopyInfo;
//	CopyInfo.Size = FIntVector(TextureSize.X, TextureSize.Y, 0);
//	AddCopyTexturePass(*GraphBuilder, OutputRDGTexture, CopyToRDGTexture, CopyInfo);
//}
//#else // !RHI_RAYTRACING
//{
//	unimplemented();
//}
//#endif
