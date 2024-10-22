#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "GridObject.generated.h"

UCLASS()
class UNREALRLLABS_API AGridObject : public AActor {
    GENERATED_BODY()

public:
    AGridObject();

    // Initializes the grid object with the specified size
    void InitializeGridObject(FVector InObjectSize);

    // Sets the grid object to be active or inactive
    void SetGridObjectActive(bool bIsActive);

    // Gets the bounds of the grid object
    FVector GetObjectExtent() const;

    // Checks if the grid object is active
    bool IsActive() const;

    // Enables or disables physics simulation for the grid object
    void SetSimulatePhysics(bool bEnablePhysics);

    // Sets the color of the grid object
    void SetGridObjectColor(FLinearColor Color);

    // Root component (non-simulating)
    UPROPERTY(VisibleAnywhere, Category = "Components")
    USceneComponent* GridObjectRoot;

    // Static mesh component for visualization
    UPROPERTY(VisibleAnywhere, Category = "Components")
    UStaticMeshComponent* MeshComponent;

private:
    // Whether the grid object is active
    bool bIsActive;

    // Dynamic material instance for changing colors
    UMaterialInstanceDynamic* DynMaterial;
};
