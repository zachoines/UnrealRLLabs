#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MainPlatform.generated.h"

UCLASS()
class UNREALRLLABS_API AMainPlatform : public AActor
{
    GENERATED_BODY()

public:
    // Sets default values for this actor's properties
    AMainPlatform();

    // Root component to allow for physics-simulating children
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    USceneComponent* RootSceneComponent;

    // Static mesh component representing the platform
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    UStaticMeshComponent* PlatformMeshComponent;

    // Initializes the platform with the specified mesh and material
    void InitializePlatform(UStaticMesh* Mesh, UMaterial* Material);
};
