// ActionSpace.cpp

#include "ActionSpace.h"

UActionSpace::UActionSpace()
    : ActionType(EActionType::Discrete)
    , NumDiscreteActions(0)
{
}

void UActionSpace::InitDiscrete(int32 NumActions)
{
    ActionType = EActionType::Discrete;
    NumDiscreteActions = NumActions;
    ContinuousActionRanges.Empty();
}

void UActionSpace::InitContinuous(const TArray<FActionRange>& Ranges)
{
    ActionType = EActionType::Continuous;
    ContinuousActionRanges = Ranges;
    NumDiscreteActions = 0;
}

TArray<float> UActionSpace::Sample() const
{
    TArray<float> SampledActions;

    if (ActionType == EActionType::Discrete)
    {
        int32 RandomIndex = FMath::RandRange(0, NumDiscreteActions - 1);
        for (int32 i = 0; i < NumDiscreteActions; i++)
        {
            SampledActions.Add(i == RandomIndex ? 1.0f : 0.0f);
        }
    }
    else if (ActionType == EActionType::Continuous)
    {
        for (const FActionRange& Range : ContinuousActionRanges)
        {
            SampledActions.Add(FMath::RandRange(Range.Min, Range.Max));
        }
    }

    return SampledActions;
}

int32 UActionSpace::GetNumActions() const
{
    if (ActionType == EActionType::Discrete)
    {
        return NumDiscreteActions;
    }
    else if (ActionType == EActionType::Continuous)
    {
        return ContinuousActionRanges.Num();
    }
    return 0;
}

EActionType UActionSpace::GetActionType() const
{
    return ActionType;
}
