// NOTICE: This file includes modifications generated with the assistance of generative AI.
// Original code structure and logic by the project author.

#include "TerraShift/GoalPlatform.h"

AGoalPlatform::AGoalPlatform()
{
    PrimaryActorTick.bCanEverTick = false;

    MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("MeshComponent"));
    RootComponent = MeshComponent;
    DynMaterial = nullptr;
    
    IsActive = true;
}

void AGoalPlatform::InitializeGoalPlatform(FVector Location, FVector Scale, FLinearColor Color, AActor* ParentPlatform)
{
    UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));
    if (PlaneMesh)
    {
        MeshComponent->SetStaticMesh(PlaneMesh);
        MeshComponent->SetMobility(EComponentMobility::Movable);
        MeshComponent->SetSimulatePhysics(false);
        MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);

        UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor"));
        if (BaseMaterial)
        {
            CurrentColor = Color;
            DynMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, this);
            DynMaterial->SetVectorParameterValue("Color", Color);
            MeshComponent->SetMaterial(0, DynMaterial);
        }

        SetActorScale3D(Scale);

        AttachToActor(ParentPlatform, FAttachmentTransformRules::KeepRelativeTransform);

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

FLinearColor AGoalPlatform::GetGoalColor() const
{
    return CurrentColor;
}
