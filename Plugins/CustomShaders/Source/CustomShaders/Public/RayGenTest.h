#pragma once

#include "../Private/RayTracing/RayTracingScene.h"
#include "../Private/SceneRendering.h"
#include "Components/ActorComponent.h"
#include "RenderGraphUtils.h"
#include "Engine/TextureRenderTargetVolume.h"
#include "Runtime/Engine/Classes/Engine/TextureRenderTarget2D.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "RenderTargetPool.h"
#include "RHI.h"
#include "RHIUtilities.h"
#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"
#include "RayTracingDefinitions.h"
#include "RayTracingPayloadType.h"
#include "RenderGraphUtils.h"
#include "SceneInterface.h"
#include "Camera/CameraComponent.h"
#include "Camera/CameraActor.h"
#include "RayGenTest.generated.h"


struct FRayGenTestParameters
{

	FIntPoint GetRenderTargetSize() const
	{
		return CachedRenderTargetSize;
	}

	FRayGenTestParameters() {}; // consider delete this, otherwise the target size will not be set, or just add setter
	FRayGenTestParameters(UTextureRenderTarget2D* IORenderTarget)
		: RenderTarget(IORenderTarget)
	{
		CachedRenderTargetSize = RenderTarget ? FIntPoint(RenderTarget->SizeX, RenderTarget->SizeY) : FIntPoint::ZeroValue;
	}

	UTextureRenderTarget2D* RenderTarget;
	FRayTracingScene* Scene;
	FIntPoint CachedRenderTargetSize;
	FTransform CameraTransform;
};

UCLASS()
class ARayGenTest : public AActor
{
public:
	GENERATED_BODY()

	ARayGenTest();
	virtual void BeginPlay() override;
	void BeginRendering();
	void EndRendering();
	void UpdateParameters(FRayGenTestParameters& DrawParameters);

	void Execute_RenderThread(FPostOpaqueRenderParameters& Parameters);

	/// The delegate handle to our function that will be executed each frame by the renderer
	FDelegateHandle PostOpaqueRenderDelegate;
	/// Cached Shader Manager Parameters
	FRayGenTestParameters CachedParams;
	/// Whether we have cached parameters to pass to the shader or not
	volatile bool bCachedParamsAreValid;

	/// We create the shader's output texture and UAV and save to avoid reallocation
	FBufferRHIRef CameraDataBuffer;
	FTexture2DRHIRef ShaderOutputTexture;
	FUnorderedAccessViewRHIRef ShaderOutputTextureUAV;
	FUnorderedAccessViewRHIRef CameraDataBufferUAV;
};