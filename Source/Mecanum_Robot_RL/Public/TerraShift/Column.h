// Column.h

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Column.generated.h"

UCLASS()
class UNREALRLLABS_API AColumn : public AActor
{
    GENERATED_BODY()

public:
    // Constructor
    AColumn();

    // Initialize the column with its scale, position, and maximum scale factor
    void InitColumn(FVector Scale, FVector Location, float MaxScaleFactor);

    // Set the column's height based on a scalar from 0 to 1
    void SetColumnHeight(float ScalarHeight);

    // Get the current height as a scalar from 0 to 1
    float GetColumnHeight() const;

    // Set the color of the column
    void SetColumnColor(FLinearColor Color);

    // Reset the column to its initial state
    void ResetColumn();

protected:
    // Called when the game starts or when the actor is spawned
    virtual void BeginPlay() override;

private:
    // The static mesh component representing the column
    UPROPERTY(VisibleAnywhere)
    UStaticMeshComponent* ColumnMesh;

    // Dynamic material instance to set the column color
    UMaterialInstanceDynamic* DynMaterial;

    // The starting scale of the column
    FVector StartingScale;

    // The maximum scale factor for the column's height
    float MaximumScaleFactor;

    // The original height of the column mesh (before scaling)
    float OriginalMeshHeight;

    // The starting location of the column
    FVector StartingLocation;

    // The current scalar height
    float CurrentScalarHeight;
};
