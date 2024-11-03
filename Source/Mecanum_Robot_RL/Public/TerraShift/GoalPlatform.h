#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/StaticMeshComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "GoalPlatform.generated.h"

UCLASS()
class UNREALRLLABS_API AGoalPlatform : public AActor
{
    GENERATED_BODY()

public:
    // Sets default values for this actor's properties
    AGoalPlatform();

    // The Static Mesh Component for the Goal Platform
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    UStaticMeshComponent* MeshComponent;

    // Initializes the Goal Platform with specified parameters
    void InitializeGoalPlatform(FVector Location, FVector Scale, FLinearColor Color, AActor* ParentPlatform);

    // Returns the relative location of the Goal Platform
    FVector GetRelativeLocation() const;

    // Sets the Goal Platform's active state
    void SetGoalPlatformActive(bool bIsActive);

    // Checks if the Goal Platform is active
    bool IsGoalPlatformActive() const;

private:
    bool IsActive;

    // Dynamic material instance for changing colors
    UMaterialInstanceDynamic* DynMaterial;
};
