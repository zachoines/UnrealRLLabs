#include "TerraShift/Column.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"
#include "Engine/StaticMesh.h"
#include "Materials/Material.h"

AColumn::AColumn() {
    PrimaryActorTick.bCanEverTick = false;

    // Create the root component (non-simulating)
    ColumnRoot = CreateDefaultSubobject<USceneComponent>(TEXT("ColumnRoot"));
    RootComponent = ColumnRoot;

    // Create the mesh component and attach it to the root
    ColumnMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("ColumnMesh"));
    ColumnMesh->SetupAttachment(ColumnRoot);

    // Disable physics simulation on the root component
    ColumnRoot->SetMobility(EComponentMobility::Movable);

    // Initialize variables
    CurrentHeight = 0.0f;
    DynMaterial = nullptr;
}

void AColumn::InitColumn(FVector Scale, FVector Location) {
    // Load the cube mesh asset
    UStaticMesh* ColumnMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
    if (ColumnMeshAsset) {
        ColumnMesh->SetStaticMesh(ColumnMeshAsset);
        ColumnMesh->SetWorldScale3D(Scale);
        ColumnMesh->SetRelativeLocation(FVector::ZeroVector);

        // Set the actor's location
        SetActorRelativeLocation(Location);
        StartingPosition = Location;

        // Load and set the dynamic material
        UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor"));
        if (BaseMaterial) {
            DynMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, this);
            ColumnMesh->SetMaterial(0, DynMaterial);
        }

        // Initialize collision settings
        ColumnMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        ColumnMesh->SetSimulatePhysics(false);
        ColumnMesh->SetEnableGravity(false);
        ColumnMesh->SetMobility(EComponentMobility::Movable);

        ResetColumn();
    }
}

bool AColumn::SetColumnHeight(float NewHeight) {
    if (!FMath::IsNearlyEqual(NewHeight, CurrentHeight)) {
        CurrentHeight = NewHeight;
        UpdateColumnPosition(NewHeight);
        return true;
    }
    return false;
}

void AColumn::ResetColumn() {
    SetColumnHeight(0.0f);
    SetSimulatePhysics(false);
    SetColumnColor(FLinearColor::White);
}

float AColumn::GetColumnHeight() const {
    return CurrentHeight;
}

void AColumn::UpdateColumnPosition(float NewHeight) {
    FVector NewPosition = StartingPosition;
    NewPosition.Z += NewHeight;
    SetActorRelativeLocation(NewPosition);
}

void AColumn::SetSimulatePhysics(bool bEnableCollision) {
    // Ensure physics simulation is disabled so the columns remain stationary
    ColumnMesh->SetSimulatePhysics(false);
    ColumnMesh->SetEnableGravity(false);

    if (bEnableCollision) {
        // Enable collision without physics simulation
        ColumnMesh->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
        ColumnMesh->SetCollisionObjectType(ECollisionChannel::ECC_WorldStatic); // Mark as static object
        ColumnMesh->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);

        // Optionally, set collision responses for specific channels if needed
        // ColumnMesh->SetCollisionResponseToChannel(ECollisionChannel::ECC_Pawn, ECollisionResponse::ECR_Block);

        SetColumnColor(FLinearColor::Blue);
    }
    else {
        // Disable collision
        ColumnMesh->SetCollisionEnabled(ECollisionEnabled::NoCollision);
        SetColumnColor(FLinearColor::White);
    }
}

void AColumn::SetColumnColor(FLinearColor Color) {
    if (DynMaterial) {
        DynMaterial->SetVectorParameterValue("Color", Color);
        ColumnMesh->SetMaterial(0, DynMaterial);
    }
}
