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
}

void AColumn::InitColumn(FVector Dimensions, FVector Location, float MaxHeight)
{
    UStaticMesh* ColumnMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));

    if (ColumnMeshAsset)
    {
        ColumnMesh->SetStaticMesh(ColumnMeshAsset);
        ColumnMesh->SetWorldScale3D(Dimensions);
        SetActorLocation(Location);
        this->MaximumHeight = MaxHeight;
        this->StartingLocation = Location;
        SetColumnHeight(0.0f);
        UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor"));
        if (BaseMaterial)
        {
            DynMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, this);
            if (DynMaterial)
            {
                ColumnMesh->SetMaterial(0, DynMaterial);
                SetColumnColor(FLinearColor(0.0f, 0.0f, 0.0f));
            }
        }
    }
}

void AColumn::SetColumnHeight(float ScalarHeight)
{
    ScalarHeight = FMath::Clamp(ScalarHeight, -1.0f, 1.0f);
    FVector NewLocation = StartingLocation;
    NewLocation.Z += ScalarHeight * MaximumHeight;
    SetActorLocation(NewLocation);
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
    SetColumnHeight(0.0f);
    SetColumnColor(FLinearColor(0.5f, 0.5f, 0.5f));
}

void AColumn::BeginPlay()
{
    Super::BeginPlay();
}

float AColumn::GetColumnHeight() const
{
    // Calculate the height as a scalar from -1 to 1
    float DeltaZ = GetActorLocation().Z - StartingLocation.Z;
    return DeltaZ / MaximumHeight;
}
