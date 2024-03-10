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
#include "TestRunner.generated.h"

UCLASS()
class ATestRunner : public AActor
{
	GENERATED_BODY()

public:
	ATestRunner();
	URayGenTest* Test;
	AStaticMeshActor* PlaneActor;
	UMaterialInstanceDynamic* DynMaterial;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = ShaderDemo)
	class UTextureRenderTarget2D* RenderTarget = nullptr;

protected:
	virtual void BeginPlay() override;
	void UpdateTestParameters();
	void InitializeRayTracer();

	float TranscurredTime; ///< allows us to add a delay on BeginPlay() 
	bool Initialized;

public:
	virtual void Tick(float DeltaTime) override;
	virtual void BeginDestroy() override;
};