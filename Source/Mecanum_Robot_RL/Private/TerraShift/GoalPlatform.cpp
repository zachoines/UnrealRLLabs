#include "TerraShift/GoalPlatform.h"

// Sets default values
AGoalPlatform::AGoalPlatform()
{
    PrimaryActorTick.bCanEverTick = false;

    // Create and set up the Static Mesh Component
    MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshComponent"));
    RootComponent = MeshComponent;
    DynMaterial = nullptr;
    
    // Initialize the active state
    IsActive = true;
}

void AGoalPlatform::InitializeGoalPlatform(FVector Location, FVector Scale, FLinearColor Color, AActor* ParentPlatform)
{
    // Load the Plane Mesh
    UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
    if (PlaneMesh)
    {
        MeshComponent->SetStaticMesh(PlaneMesh);
        MeshComponent->SetMobility(EComponentMobility::Movable);
        MeshComponent->SetSimulatePhysics(false);
        MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);

        // Load and set the dynamic material
        UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor"));
        if (BaseMaterial)
        {
            DynMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, this);
            DynMaterial->SetVectorParameterValue("Color", Color);
            MeshComponent->SetMaterial(0, DynMaterial);
        }

        // Set the scale
        SetActorScale3D(Scale);

        // Attach to the main platform
        AttachToActor(ParentPlatform, FAttachmentTransformRules::KeepRelativeTransform);

        // Set relative location
        SetActorRelativeLocation(Location);
    }
}

FVector AGoalPlatform::GetRelativeLocation() const
{
    return MeshComponent->GetRelativeLocation();
}

void AGoalPlatform::SetGoalPlatformActive(bool bIsActive)
{
    IsActive = bIsActive;
    SetActorHiddenInGame(!bIsActive);
    SetActorEnableCollision(bIsActive);
}

bool AGoalPlatform::IsGoalPlatformActive() const
{
    return IsActive;
}
