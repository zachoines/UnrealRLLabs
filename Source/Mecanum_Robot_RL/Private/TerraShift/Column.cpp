#include "TerraShift/Column.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"

AColumn::AColumn()
{
    PrimaryActorTick.bCanEverTick = false;

    ColumnRoot = CreateDefaultSubobject<USceneComponent>(TEXT("ColumnRoot"));
    RootComponent = ColumnRoot;

    ColumnMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("ColumnMesh"));
    ColumnMesh->SetupAttachment(ColumnRoot);

    ColumnRoot->SetMobility(EComponentMobility::Movable);

    CurrentHeight = 0.0f;
    DynMaterial = nullptr;
}

void AColumn::InitColumn(FVector Scale, FVector Location)
{
    UStaticMesh* ColumnMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Script/Engine.StaticMesh'/Game/Shapes/Column.Column'"));
    if (ColumnMeshAsset) {
        ColumnMesh->SetStaticMesh(ColumnMeshAsset);
        ColumnMesh->SetWorldScale3D(Scale);
        ColumnMesh->SetRelativeLocation(FVector::ZeroVector);

        SetActorRelativeLocation(Location);
        StartingPosition = Location;

        UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Script/Engine.Material'/Game/Material/Column_Material.Column_Material'"));
        if (BaseMaterial) {
            DynMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, this);
            ColumnMesh->SetMaterial(0, DynMaterial);
        }

        ColumnMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        ColumnMesh->SetSimulatePhysics(false);
        ColumnMesh->SetEnableGravity(false);
        ColumnMesh->SetMobility(EComponentMobility::Movable);

        ResetColumn(0.0);
    }
}

bool AColumn::SetColumnHeight(float NewHeight)
{
    if (!FMath::IsNearlyEqual(NewHeight, CurrentHeight)) {
        CurrentHeight = NewHeight;
        UpdateColumnPosition(NewHeight);
        return true;
    }
    return false;
}

void AColumn::ResetColumn(float height)
{
    SetColumnHeight(height);
    SetSimulatePhysics(false);
    SetColumnColor(FLinearColor::White);
}

float AColumn::GetColumnHeight() const
{
    return CurrentHeight;
}

void AColumn::UpdateColumnPosition(float NewHeight)
{
    FVector NewPosition = StartingPosition;
    NewPosition.Z += NewHeight;
    SetActorRelativeLocation(NewPosition);
}

void AColumn::SetSimulatePhysics(bool bEnableCollision)
{
    ColumnMesh->SetSimulatePhysics(false);
    ColumnMesh->SetEnableGravity(false);

    if (bEnableCollision) {
        ColumnMesh->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        ColumnMesh->SetCollisionObjectType(ECollisionChannel::ECC_WorldStatic);
        ColumnMesh->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
    }
    else {
        ColumnMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    }
}

void AColumn::SetColumnColor(FLinearColor Color)
{
    if (DynMaterial) {
        DynMaterial->SetVectorParameterValue("Color", Color);
        ColumnMesh->SetMaterial(0, DynMaterial);
    }
}
