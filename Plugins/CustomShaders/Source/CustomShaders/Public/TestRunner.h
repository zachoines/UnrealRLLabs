#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RayGenTest.h"
#include "Engine/World.h"
#include "SceneInterface.h"
#include "../Private/ScenePrivate.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/StaticMesh.h"
#include "Components/SceneCaptureComponent2D.h"  // Added for SceneCaptureComponent2D

#include "CoreMinimal.h"
#include "Camera/CameraComponent.h"
#include "RendererInterface.h"
#include "RenderGraphUtils.h"

#include "TestRunner.generated.h"

UCLASS()
class ATestRunner : public AActor
{
    GENERATED_BODY()

public:
    ATestRunner();
    ARayGenTest* Test;
    AStaticMeshActor* PlaneActor;
    UMaterialInstanceDynamic* DynMaterial;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = ShaderDemo)
    class UTextureRenderTarget2D* RenderTarget = nullptr;

    // Dedicated SceneCaptureComponent2D for the static camera
    UPROPERTY(VisibleAnywhere, BlueprintReadWrite, Category = "Ray Tracing")
    USceneCaptureComponent2D* StaticCaptureComponent;

protected:
    virtual void BeginPlay() override;
    void UpdateTestParameters();
    void InitializeRayTracer();

    float TranscurredTime;
    bool Initialized;

public:
    virtual void Tick(float DeltaTime) override;
    virtual void BeginDestroy() override;
};
