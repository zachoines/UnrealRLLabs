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
    /** Sets default values for this actor's properties. */
    AGoalPlatform();

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
    UStaticMeshComponent* MeshComponent;

    /** Initializes the goal platform mesh, scale, color, and parent. */
    void InitializeGoalPlatform(FVector Location, FVector Scale, FLinearColor Color, AActor* ParentPlatform);

    /** Returns the relative location of the goal platform. */
   FVector GetRelativeLocation() const;

    /** Enables or disables the goal platform. */
    void SetGoalPlatformActive(bool bIsActive);

    /** Returns true when the goal platform is active. */
    bool IsGoalPlatformActive() const;

    /** Returns the current material color. */
    FLinearColor GetGoalColor() const;

private:
    bool IsActive;

    UMaterialInstanceDynamic* DynMaterial;

    FLinearColor CurrentColor;
};
