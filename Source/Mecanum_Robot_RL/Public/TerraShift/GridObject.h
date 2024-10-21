#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "GridObject.generated.h"

UCLASS()
class UNREALRLLABS_API AGridObject : public AActor {
    GENERATED_BODY()

public:
    AGridObject();

    // Initialize the grid object with a mesh and size
    void InitializeGridObject(FVector InObjectSize);

    // Set the grid object to be active or inactive
    void SetGridObjectActive(bool bIsActive);

    // Get the bounds of the grid object
    FVector GetObjectExtent() const;

    // Check if the grid object is active
    bool IsActive() const;

    // Static mesh component for visualization
    UPROPERTY(VisibleAnywhere)
    UStaticMeshComponent* MeshComponent;

private:
    
    // Whether the grid object is active
    bool bIsActive;
};
