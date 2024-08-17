#include "TestRunner.h"
#include "Camera/CameraActor.h" // Not used anymore, but included for clarity

ATestRunner::ATestRunner()
{
    PrimaryActorTick.bCanEverTick = true;
    TranscurredTime = 0.0f;
    Initialized = false;

    // Create and initialize the SceneCaptureComponent2D
    StaticCaptureComponent = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("StaticCaptureComponent"));
    StaticCaptureComponent->SetupAttachment(RootComponent);
    StaticCaptureComponent->bUseRayTracingIfEnabled = true;
    StaticCaptureComponent->Activate(true);

    // **IMPORTANT:** Set the TextureTarget to your RenderTarget in the editor or Blueprint.
    // Uncomment the line below if you want to set it in C++ directly.
    StaticCaptureComponent->TextureTarget = RenderTarget; 

}

void ATestRunner::BeginPlay()
{
    Super::BeginPlay();

    // Load the plane mesh
    UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));

    // Spawn the StaticMeshActor for the plane
    FActorSpawnParameters SpawnParams;
    SpawnParams.Owner = this;
    FVector Location(0.f, 200.f, 0.f);
    FRotator Rotation(0, 0, 90);

    PlaneActor = GetWorld()->SpawnActor<AStaticMeshActor>(Location, Rotation, SpawnParams);
    PlaneActor->GetStaticMeshComponent()->SetStaticMesh(PlaneMesh);
    PlaneActor->GetStaticMeshComponent()->SetWorldScale3D(FVector3d(1.0, 1.0, 1));

    UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("Material'/Game/Material/Ray_Cast_View.Ray_Cast_View'"));
    BaseMaterial->TwoSided = true;
    PlaneActor->GetStaticMeshComponent()->SetMaterial(0, BaseMaterial);

    // Get the material from the PlaneMesh, assuming it has one
    UMaterialInterface* PlaneMaterial = PlaneActor->GetStaticMeshComponent()->GetMaterial(0);

    // Create a dynamic material instance from the PlaneMesh's material
    DynMaterial = UMaterialInstanceDynamic::Create(PlaneMaterial, this);

    // Apply the dynamic material to the static mesh component of the PlaneActor
    PlaneActor->GetStaticMeshComponent()->SetMaterial(0, DynMaterial);

    PlaneActor->GetStaticMeshComponent()->SetEnableGravity(false);
}

void ATestRunner::InitializeRayTracer()
{
    Test = NewObject<ARayGenTest>(this);

    Initialized = true;

    UpdateTestParameters();
}

void ATestRunner::UpdateTestParameters()
{
    FRayGenTestParameters parameters;
    parameters.Scene = &GetWorld()->Scene->GetRenderScene()->RayTracingScene;
    parameters.CachedRenderTargetSize = FIntPoint(RenderTarget->SizeX, RenderTarget->SizeY);
    parameters.RenderTarget = RenderTarget;

    if (StaticCaptureComponent)
    {
        parameters.CameraTransform = StaticCaptureComponent->GetComponentTransform();
    }

    Test->UpdateParameters(parameters);
}

// Called every frame
void ATestRunner::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // We want a slight delay before we start, otherwise some resources such as the accelerated structure will not be ready
    if (RenderTarget != nullptr && TranscurredTime > 3.0f)
    {
        if (Initialized)
        {
            UpdateTestParameters();
        }
        else
        {
            InitializeRayTracer();
            ENQUEUE_RENDER_COMMAND(BeginRenderingCommand)(
                [this](FRHICommandListImmediate& RHICmdList)
                {
                    Test->BeginRendering();
                });
        }
    }
    else
    {
        TranscurredTime += DeltaTime;
    }
}

void ATestRunner::BeginDestroy()
{
    Super::BeginDestroy();
    if (Initialized)
    {
        Test->EndRendering();
    }
}
