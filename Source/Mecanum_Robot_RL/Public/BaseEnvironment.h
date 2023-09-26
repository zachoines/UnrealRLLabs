#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ActionSpace.h"
#include "RLTypes.h"
#include "BaseEnvironment.generated.h"

USTRUCT(BlueprintType)
struct MECANUM_ROBOT_RL_API FState
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadWrite, Category = "State")
    TArray<float> Values;
};

USTRUCT(BlueprintType)
struct MECANUM_ROBOT_RL_API FBaseInitParams
{
    GENERATED_USTRUCT_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment Params")
    FVector Location;
};

UCLASS()
class MECANUM_ROBOT_RL_API ABaseEnvironment : public AActor
{
    GENERATED_BODY()

public:

    // Sets default values for this actor's properties
    ABaseEnvironment();

    // Initialize the environment using the FBaseInitParams struct
    virtual void InitEnv(FBaseInitParams* Params);

    // Reset the environment and return the initial state
    virtual FState ResetEnv();

    // Perform a step in the environment using the given action
    virtual TTuple<bool, float, FState> Step(FAction Action);

    virtual void PostInitializeComponents() override;

protected:
    // Calculate the reward for the current state
    virtual float Reward();

public:
    UPROPERTY(BlueprintReadOnly, Category = "Action")
    UActionSpace* ActionSpace;  // Public member variable for the ActionSpace
};
