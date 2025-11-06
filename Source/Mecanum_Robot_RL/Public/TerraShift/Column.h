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

    /** Initializes the column with the given scale and location. */
    void InitColumn(FVector Scale, FVector Location);

    /** Sets the column height and updates its position. */
    bool SetColumnHeight(float NewHeight);

    /** Resets the column to its initial state. */
    void ResetColumn(float height);

    /** Returns the current column height. */
    float GetColumnHeight() const;

    /** Enables or disables physics simulation for the column. */
    void SetSimulatePhysics(bool bEnablePhysics);

    /** Sets the column color. */
    void SetColumnColor(FLinearColor Color);

    UPROPERTY(VisibleAnywhere, Category = "Components")
    USceneComponent* ColumnRoot;

    UPROPERTY(VisibleAnywhere, Category = "Components")
    UStaticMeshComponent* ColumnMesh;

private:
    float CurrentHeight;

    FVector StartingPosition;

    UMaterialInstanceDynamic* DynMaterial;

    /** Recomputes the column's world-space position based on height. */
    void UpdateColumnPosition(float NewHeight);
};
