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
    void InitializeGridObject(FVector InObjectSize, float InObjectMass);

    // Sets the grid object to be active or inactive
    void SetGridObjectActive(bool bIsActive);

    // Gets the world bounds of the grid object
    FVector GetObjectExtent() const;

    // Gets the world location of the grid object
    FVector GetObjectLocation() const;

    // Checks if the grid object is active
    bool IsActive() const;

    // Enables or disables physics simulation for the grid object
    void SetSimulatePhysics(bool bEnablePhysics);

    // Sets the color of the grid object
    void SetGridObjectColor(FLinearColor Color);

    // Reset the GridObject
    void ResetGridObject();

    // Static mesh component for visualization (now the root component)
    UPROPERTY(VisibleAnywhere, Category = "Components")
    UStaticMeshComponent* MeshComponent;

private:
    // Whether the grid object is active
    bool bIsActive;

    // Dynamic material instance for changing colors
    UMaterialInstanceDynamic* DynMaterial;
};
