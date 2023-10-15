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
//private:
//    FState CurrentState;
//    FState LastState;
//    FAction LastAction;

public:

    // Sets default values for this actor's properties
    ABaseEnvironment();

    UPROPERTY(BlueprintReadOnly, Category = "Action")
    UActionSpace* ActionSpace;  // Public member variable for the ActionSpace

    // Initialize the environment using the FBaseInitParams struct
    virtual void InitEnv(FBaseInitParams* Params);

    // Reset the environment and return the initial state
    virtual FState ResetEnv();

    virtual void PostInitializeComponents() override;


    // Update environment with actions
    virtual void Act(FAction Action);
  
    /* 
        Optional convenience function. Will be called before State(), Done(), Trunc(), and Reward().
    */
    virtual void Update();

    // Returns the public view of the state. Called after Update 
    virtual FState State();

    // Returns done conditon
    virtual bool Done();

    // Returns truncation conditon
    virtual bool Trunc();

    // Calculate the reward for the current state
    virtual float Reward();

};
