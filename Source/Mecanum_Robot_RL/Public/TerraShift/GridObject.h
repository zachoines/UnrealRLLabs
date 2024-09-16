#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "string"
#include "GridObject.generated.h"

UCLASS()
class UNREALRLLABS_API AGridObject : public AActor
{
    GENERATED_BODY()

public:
    AGridObject();

    // Function to initialize the grid object with a mesh, material and size
    void InitializeGridObject(FVector InObjectSize, UStaticMesh* Mesh, UMaterial* Material);

    // Function to activate or deactivate the grid object
    void SetGridObjectActive(bool bIsActive);

    // Function to set the location and activate the grid object
    void SetActorLocationAndActivate(FVector NewLocation);

    // New function to get the bounds of the object's mesh
    FVector GetObjectExtent() const;

private:
    UPROPERTY(VisibleAnywhere)
    UStaticMesh* ObjectMesh;
    UStaticMeshComponent* MeshComponent;
    UMaterial* ObjectMaterial;
    FVector ObjectSize;
};
