#include "TerraShift/Column.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "GameFramework/Actor.h"

AColumn::AColumn()
{
    PrimaryActorTick.bCanEverTick = true;

    ColumnMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("ColumnMesh"));
    RootComponent = ColumnMesh;

    ColumnMesh->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
    ColumnMesh->SetMobility(EComponentMobility::Movable);
    ColumnMesh->SetEnableGravity(false);
    ColumnMesh->SetSimulatePhysics(false);
}

void AColumn::InitColumn(FVector Dimensions, FVector Location, float MaxHeight)
{

    UStaticMesh* ColumnMeshAsset = LoadObject<UStaticMesh>(nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));

    if (ColumnMeshAsset)
    {
        ColumnMesh->SetStaticMesh(ColumnMeshAsset);
        ColumnMesh->SetWorldScale3D(Dimensions);

        // Set collision, mobility, and physics properties
        ColumnMesh->SetCollisionEnabled(ECollisionEnabled::QueryOnly);
        ColumnMesh->SetMobility(EComponentMobility::Movable);
        ColumnMesh->BodyInstance.SetMassOverride(0.01f, true);
        ColumnMesh->SetEnableGravity(false);
        ColumnMesh->SetSimulatePhysics(false);

        SetActorLocation(Location);
        this->MaximumHeight = MaxHeight;
        SetColumnHeight(0.5f);
        SetColumnColor(FLinearColor(1.0f, 1.0f, 1.0f));
    }
}

void AColumn::SetColumnHeight(float NewHeight)
{
    FVector CurrentLocation = GetActorLocation();
    CurrentLocation.Z = FMath::Clamp(NewHeight * MaximumHeight, 0.0f, MaximumHeight);
    SetActorLocation(CurrentLocation);
}

void AColumn::SetColumnColor(FLinearColor Color)
{
    // Create a dynamic material instance to set the column color
    UMaterialInstanceDynamic* DynMaterial = UMaterialInstanceDynamic::Create(ColumnMesh->GetMaterial(0), this);
    if (DynMaterial)
    {
        DynMaterial->SetVectorParameterValue("Color", Color);
        ColumnMesh->SetMaterial(0, DynMaterial);
    }
}

void AColumn::SetColumnAcceleration(float Acceleration)
{
    // Update the velocity of the column by adding the acceleration
    Velocity += Acceleration;
}

void AColumn::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // Update the column's position based on its velocity
    FVector CurrentLocation = GetActorLocation();
    CurrentLocation.Z += Velocity * DeltaTime;
    CurrentLocation.Z = FMath::Clamp(CurrentLocation.Z, 0.0f, MaximumHeight);

    // If the column hits the maximum or minimum height, stop its movement
    if (CurrentLocation.Z == 0.0f || CurrentLocation.Z == MaximumHeight)
    {
        Velocity = 0.0f;
    }

    SetActorLocation(CurrentLocation);
}

void AColumn::ResetColumn()
{
    Velocity = 0.0f;
    SetColumnHeight(0.5f);
    SetColumnColor(FLinearColor(1.0f, 1.0f, 1.0f));
}

void AColumn::BeginPlay()
{
    Super::BeginPlay();

    if (ColumnMesh)
    {
        UMaterial* BaseMaterial = LoadObject<UMaterial>(nullptr, TEXT("/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor"));
        if (BaseMaterial)
        {
            ColumnMesh->SetMaterial(0, BaseMaterial);
        }
    }
}

float AColumn::GetColumnHeight() const
{
    // Get the current height as a percentage of MaxHeight
    return GetActorLocation().Z / MaximumHeight;
}
