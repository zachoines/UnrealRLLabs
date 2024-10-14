// Column.cpp

#include "TerraShift/Column.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "GameFramework/Actor.h"

AColumn::AColumn()
{
    PrimaryActorTick.bCanEverTick = false;

    ColumnMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("ColumnMesh"));
    DynMaterial = nullptr;
    RootComponent = ColumnMesh;

    ColumnMesh->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    ColumnMesh->SetCollisionResponseToAllChannels(ECollisionResponse::ECR_Block);
    ColumnMesh->SetMobility(EComponentMobility::Movable);

    CurrentScalarHeight = -1.0f; // Initialize to an invalid value to ensure the first update occurs
}

void AColumn::InitColumn(FVector Scale, FVector Location, float MaxScaleFactor)
{
    UStaticMesh* ColumnMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));

    if (ColumnMeshAsset)
    {
        ColumnMesh->SetStaticMesh(ColumnMeshAsset);
        this->OriginalMeshHeight = ColumnMeshAsset->GetBounds().BoxExtent.Z * 2.0f;
        SetActorScale3D(Scale);
        SetActorLocation(Location);
        this->StartingScale = Scale;
        this->MaximumScaleFactor = MaxScaleFactor;
        this->StartingLocation = Location;
        this->CurrentScalarHeight = -1.0f; // Ensure the first call to SetColumnHeight updates the column
        SetColumnHeight(0.0f); // Start at minimum height
        UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor"));
        if (BaseMaterial)
        {
            DynMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, this);
            if (DynMaterial)
            {
                ColumnMesh->SetMaterial(0, DynMaterial);
                SetColumnColor(FLinearColor(0.5f, 0.5f, 0.5f));
            }
        }
    }
}

void AColumn::SetColumnHeight(float ScalarHeight)
{
    ScalarHeight = FMath::Clamp(ScalarHeight, 0.0f, 1.0f); // Ensure scalar is between 0 and 1

    // Only update if ScalarHeight has changed
    if (!FMath::IsNearlyEqual(ScalarHeight, CurrentScalarHeight))
    {
        CurrentScalarHeight = ScalarHeight;

        float NewScaleZ = StartingScale.Z * (1.0f + ScalarHeight * (MaximumScaleFactor - 1.0f));
        FVector NewScale = StartingScale;
        NewScale.Z = NewScaleZ;

        // Adjust location to keep the bottom at the same position
        float OriginalHeight = OriginalMeshHeight * StartingScale.Z;
        float NewHeight = OriginalMeshHeight * NewScaleZ;
        float DeltaHeight = (NewHeight - OriginalHeight) / 2.0f;

        FVector NewLocation = StartingLocation;
        NewLocation.Z += DeltaHeight;

        SetActorScale3D(NewScale);
        SetActorLocation(NewLocation);
    }
}

void AColumn::SetColumnColor(FLinearColor Color)
{
    if (DynMaterial)
    {
        DynMaterial->SetVectorParameterValue("Color", Color);
        ColumnMesh->SetMaterial(0, DynMaterial);
    }
}

void AColumn::ResetColumn()
{
    CurrentScalarHeight = -1.0f; // Reset current scalar height to ensure update
    SetColumnHeight(0.0f); // Reset to minimum height
    SetColumnColor(FLinearColor(0.5f, 0.5f, 0.5f)); // Default color
}

void AColumn::BeginPlay()
{
    Super::BeginPlay();
}

float AColumn::GetColumnHeight() const
{
    return CurrentScalarHeight;
}
