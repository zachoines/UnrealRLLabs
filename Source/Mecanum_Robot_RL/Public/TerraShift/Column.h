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

    // Initialize the column with its size, position, and maximum height
    void InitColumn(FVector Dimensions, FVector Location, float MaxHeight);

    // Set the column's height based on a scalar from -1 to 1
    void SetColumnHeight(float ScalarHeight);

    // Get the current height as a scalar from -1 to 1
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

    // The maximum height the column can reach
    float MaximumHeight;

    // The starting position (location) of the column
    FVector StartingLocation;
};
