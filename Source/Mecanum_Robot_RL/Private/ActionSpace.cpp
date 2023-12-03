// ActionSpace.cpp

#include "ActionSpace.h"

UActionSpace::UActionSpace()
{

}

void UActionSpace::Init(const TArray<FContinuousActionSpec>& InContinuousActions,
    const TArray<FDiscreteActionSpec>& InDiscreteActions)
{
    ContinuousActions = InContinuousActions;
    DiscreteActions = InDiscreteActions;
}

int UActionSpace::TotalActions() 
{
    return ContinuousActions.Num() + DiscreteActions.Num();
}