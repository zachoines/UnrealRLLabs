// ActionSpace.h

#pragma once

#include "CoreMinimal.h"
#include "ActionSpace.generated.h"

UENUM(BlueprintType)
enum class EActionType : uint8
{
    Discrete,
    Continuous
};

USTRUCT(BlueprintType)
struct FActionRange
{
    GENERATED_USTRUCT_BODY()

        UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Action")
        float Min;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Action")
        float Max;
};

UCLASS()
class MECANUM_ROBOT_RL_API UActionSpace : public UObject
{
    GENERATED_BODY()

public:
    UActionSpace();

    // Initialize the action space with discrete actions
    void InitDiscrete(int32 NumActions);

    // Initialize the action space with continuous actions
    void InitContinuous(const TArray<FActionRange>& Ranges);

    // Sample a random action
    TArray<float> Sample() const;

    // Get the number of actions
    int32 GetNumActions() const;

    // Get the type of actions (Discrete or Continuous)
    EActionType GetActionType() const;

private:
    EActionType ActionType;
    int32 NumDiscreteActions;
    TArray<FActionRange> ContinuousActionRanges;
};
