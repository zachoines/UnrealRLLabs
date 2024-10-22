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

    // Initializes the column with a given scale and location
    void InitColumn(FVector Scale, FVector Location);

    // Sets the column's height and updates its position
    bool SetColumnHeight(float NewHeight);

    // Resets the column to its initial state
    void ResetColumn();

    // Retrieves the current height of the column
    float GetColumnHeight() const;

    // Enables or disables physics simulation for the column
    void SetSimulatePhysics(bool bEnablePhysics);

    // Sets the color of the column
    void SetColumnColor(FLinearColor Color);

    // The root component for the column (non-simulating)
    UPROPERTY(VisibleAnywhere, Category = "Components")
    USceneComponent* ColumnRoot;

    // The static mesh component representing the column
    UPROPERTY(VisibleAnywhere, Category = "Components")
    UStaticMeshComponent* ColumnMesh;

private:
    // The current height of the column
    float CurrentHeight;

    // The starting position of the column
    FVector StartingPosition;

    // Dynamic material instance for changing colors
    UMaterialInstanceDynamic* DynMaterial;

    // Updates the column's position based on its height
    void UpdateColumnPosition(float NewHeight);
};
