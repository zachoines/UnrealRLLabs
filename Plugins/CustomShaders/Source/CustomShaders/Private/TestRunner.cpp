#include "TestRunner.h"

ATestRunner::ATestRunner()
{
	PrimaryActorTick.bCanEverTick = true;
}

// Called when the game starts or when spawned
void ATestRunner::BeginPlay()
{
	Super::BeginPlay();
	// Load the plane mesh
	UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));

	// Spawn the StaticMeshActor for the plane
	FActorSpawnParameters SpawnParams;
	SpawnParams.Owner = this;
	FVector Location(0.0f, 0.0f, 0.0f);
	FRotator Rotation(0.0f, 0.0f, 90.0f);

	PlaneActor = GetWorld()->SpawnActor<AStaticMeshActor>(Location, Rotation, SpawnParams);
	PlaneActor->GetStaticMeshComponent()->SetStaticMesh(PlaneMesh);
	PlaneActor->GetStaticMeshComponent()->SetWorldScale3D(FVector3d(2, 2, 2));

	UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/Material/Ray_Cast_View.Ray_Cast_View'"));
	BaseMaterial->TwoSided = true;
	PlaneActor->GetStaticMeshComponent()->SetMaterial(0, BaseMaterial);

	// Get the material from the PlaneMesh, assuming it has one
	UMaterialInterface* PlaneMaterial = PlaneActor->GetStaticMeshComponent()->GetMaterial(0);

	// Create a dynamic material instance from the PlaneMesh's material
	DynMaterial = UMaterialInstanceDynamic::Create(PlaneMaterial, this);

	// Set the Render Target as the texture parameter on your material instance
	// DynMaterial->SetTextureParameterValue(TEXT("TextureParameterName"), RenderTarget);

	// Apply the dynamic material to the static mesh component of the PlaneActor
	PlaneActor->GetStaticMeshComponent()->SetMaterial(0, DynMaterial);

	PlaneActor->GetStaticMeshComponent()->SetEnableGravity(false);
}

void ATestRunner::InitializeRayTracer()
{
	Test = NewObject<URayGenTest>(this);
	
	Initialized = true;

	UpdateTestParameters();
}

void ATestRunner::UpdateTestParameters()
{
	FRayGenTestParameters parameters;
	parameters.Scene = &GetWorld()->Scene->GetRenderScene()->RayTracingScene;
	parameters.CachedRenderTargetSize = FIntPoint(RenderTarget->SizeX, RenderTarget->SizeY);
	parameters.RenderTarget = RenderTarget;
	Test->UpdateParameters(parameters);
}

// Called every frame
void ATestRunner::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);


	// we want a slight delay before we start, otherwise some resources such as the accelerated structure will not be ready
	if (RenderTarget != nullptr && TranscurredTime > 1.0f)
	{
		if (Initialized)
		{
			UpdateTestParameters();
		}
		else {
			InitializeRayTracer();
			ENQUEUE_RENDER_COMMAND(BeginRenderingCommand)(
			[this](FRHICommandListImmediate& RHICmdList)
			{
				// This will be executed on the rendering thread
				Test->BeginRendering();
			});
		}
	}
	else {
		TranscurredTime += DeltaTime;
	}

	// Update the dynamic material with the latest render target
	/*if (DynMaterial && RenderTarget)
	{
		DynMaterial->SetTextureParameterValue(TEXT("TextureParameterName"), RenderTarget);
	}*/
}

void ATestRunner::BeginDestroy()
{
	Super::BeginDestroy();

	if (Initialized) {
		Test->EndRendering();
	}
}