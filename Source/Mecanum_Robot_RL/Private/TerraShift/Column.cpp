#include "TerraShift/Column.h"

AColumn::AColumn() {
    PrimaryActorTick.bCanEverTick = false;

    ColumnMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("ColumnMesh"));
    RootComponent = ColumnMesh;
    SetSimulatePhysics(false);
    CurrentHeight = 0.0f;
}

// Toggle fo Physics
void AColumn::SetSimulatePhysics(bool bEnablePhysics) {
    SetActorEnableCollision(bEnablePhysics);
    if (bEnablePhysics) {
        // Enable collision and physics
        ColumnMesh->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        ColumnMesh->SetCollisionObjectType(ECollisionChannel::ECC_WorldStatic);
        ColumnMesh->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
        ColumnMesh->SetCollisionResponseToChannel(ECollisionChannel::ECC_Pawn, ECollisionResponse::ECR_Block);
        ColumnMesh->SetSimulatePhysics(true);
        ColumnMesh->SetEnableGravity(true);
        ColumnMesh->SetMobility(EComponentMobility::Movable);
    }
    else {
        // Disable collision and physics
        ColumnMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        ColumnMesh->SetSimulatePhysics(false);
        ColumnMesh->SetEnableGravity(false);
        ColumnMesh->SetMobility(EComponentMobility::Movable);
    }
}

void AColumn::InitColumn(FVector Scale, FVector Location) {
    UStaticMesh* ColumnMeshAsset = LoadObject<UStaticMesh>(this, TEXT("/Engine/BasicShapes/Cube.Cube"));
    if (ColumnMeshAsset) {
        ColumnMesh->SetStaticMesh(ColumnMeshAsset);
        SetActorScale3D(Scale);
        SetActorRelativeLocation(Location);
        this->StartingPosition = Location;

        // Set dynamic material
        UMaterial* BaseMaterial = LoadObject<UMaterial>(this, TEXT("/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor"));
        if (BaseMaterial) {
            DynMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, this);
            ColumnMesh->SetMaterial(0, DynMaterial);
        }

        ResetColumn();
    }
}

bool AColumn::SetColumnHeight(float NewHeight) {
    if (!FMath::IsNearlyEqual(NewHeight, CurrentHeight)) {
        CurrentHeight = NewHeight;
        UpdateColumnPosition(NewHeight);
        return true;
    }
    else {
        return false;
    }
}

void AColumn::ResetColumn() {
    SetColumnHeight(0.0);
}

float AColumn::GetColumnHeight() const {
    return CurrentHeight;
}

void AColumn::UpdateColumnPosition(float NewHeight) {
    FVector NewPosition = StartingPosition;
    NewPosition.Z += NewHeight;
    SetActorRelativeLocation(NewPosition);
}

void AColumn::SetColumnColor(FLinearColor Color)
{
    if (DynMaterial)
    {
        DynMaterial->SetVectorParameterValue("Color", Color);
        ColumnMesh->SetMaterial(0, DynMaterial);
    }
}


