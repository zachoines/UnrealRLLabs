#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Column.generated.h"

UCLASS()
class UNREALRLLABS_API AColumn : public AActor {
    GENERATED_BODY()

public:
    AColumn();

    // Initialize the column with its position and maximum height
    void InitColumn(FVector Scale, FVector Location);

    // Set the column's height directly in local space
    bool SetColumnHeight(float NewHeight);

    // Get the current column height
    float GetColumnHeight() const;

    // Reset the column to its default height
    void ResetColumn();

    // Toggle for Physics/Collisions
    void SetSimulatePhysics(bool bEnablePhysics);

    // The static mesh component representing the column
    UPROPERTY(VisibleAnywhere)
    UStaticMeshComponent* ColumnMesh;

    // Set the color of the column
    void SetColumnColor(FLinearColor Color);

private:

    // The current height of the column
    float CurrentHeight;

    // The starting position of the column
    FVector StartingPosition;

    // Dynamic material instance to set the column color
    UMaterialInstanceDynamic* DynMaterial;

    // Helper function to update the mesh position
    void UpdateColumnPosition(float fNewHeight);
};
