#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "GridObject.generated.h"

UCLASS()
class UNREALRLLABS_API AGridObject : public AActor {
    GENERATED_BODY()

public:
    AGridObject();

    /** Initializes the grid object with the specified size. */
    void InitializeGridObject(FVector InObjectSize, float InObjectMass);

    /** Enables or disables the grid object. */
    void SetGridObjectActive(bool bIsActive);

    /** Returns the world-space extent of the object. */
    FVector GetObjectExtent() const;

    /** Returns the object's world-space location. */
    FVector GetObjectLocation() const;

    /** Returns true when the grid object is active. */
    bool IsActive() const;

    /** Enables or disables physics simulation. */
    void SetSimulatePhysics(bool bEnablePhysics);

    /** Applies a color tint to the object material. */
    void SetGridObjectColor(FLinearColor Color);

    /** Resets the grid object to its default state. */
    void ResetGridObject();

    UPROPERTY(VisibleAnywhere, Category = "Components")
    UStaticMeshComponent* MeshComponent;

private:
    bool bIsActive;

    UMaterialInstanceDynamic* DynMaterial;
};
