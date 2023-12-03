#include "TestRunner.h"
#include "Engine/World.h"
#include "SceneInterface.h"
#include "../Private/ScenePrivate.h"

ATestRunner::ATestRunner()
{
	PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void ATestRunner::BeginPlay()
{
	Super::BeginPlay();
	Test = FRayGenTest();
	Initialized = false;

	if (RenderTarget != nullptr)
		UpdateTestParameters();
}

void ATestRunner::UpdateTestParameters()
{
	FRayGenTestParameters parameters;
	// parameters.Scene = &GetWorld()->Scene->GetRenderScene()->RayTracingScene;
	// parameters.CachedRenderTargetSize = FIntPoint(RenderTarget->SizeX, RenderTarget->SizeY);
	parameters.RenderTarget = RenderTarget;
	Test.UpdateParameters(parameters);
}

// Called every frame
void ATestRunner::Tick(float DeltaTime)
{
	TranscurredTime+=DeltaTime;
	Super::Tick(DeltaTime);

	// we want a slight delay before we start, otherwise some resources such as the accelerated structure will not be ready
	if(RenderTarget != nullptr && TranscurredTime>1.0f)
	{
		UpdateTestParameters();

		if(!Initialized)
		{
			// Test.BeginRendering();
			Initialized = true;
		}
	}
}
void ATestRunner::BeginDestroy()
{
	Super::BeginDestroy();
	// Test.EndRendering();
}