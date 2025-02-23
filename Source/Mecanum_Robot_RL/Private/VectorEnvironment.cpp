//#include "VectorEnvironment.h"
//#include "EnvironmentConfig.h"  // So we can call EnvConfig->Get(...), HasPath(...), etc.
//#include "Misc/Paths.h"
//#include "BaseEnvironment.h"
//
//// Sets default values
//AVectorEnvironment::AVectorEnvironment()
//{
//    PrimaryActorTick.bCanEverTick = false;
//    CurrentAgents = 1;
//}
//
//void AVectorEnvironment::BeginPlay()
//{
//}
//
//void AVectorEnvironment::InitEnv(
//    TSubclassOf<ABaseEnvironment> EnvironmentClass,
//    TArray<FBaseInitParams*> ParamsArray
//)
//{
//    // Safety check
//    if (ParamsArray.Num() == 0)
//    {
//        UE_LOG(LogTemp, Warning, TEXT("AVectorEnvironment::InitEnv - No init params provided."));
//        return;
//    }
//
//    // We assume the first param has a pointer to the UEnvironmentConfig
//    FBaseInitParams* FirstParam = ParamsArray[0];
//    if (!FirstParam->EnvConfig)
//    {
//        UE_LOG(LogTemp, Error, TEXT("AVectorEnvironment::InitEnv - EnvConfig is null in first param!"));
//        return;
//    }
//
//    // Store a pointer to the environment config
//    EnvConfig = FirstParam->EnvConfig;
//
//    // Spawn each environment
//    for (FBaseInitParams* Param : ParamsArray)
//    {
//        ABaseEnvironment* Env = GetWorld()->SpawnActor<ABaseEnvironment>(
//            EnvironmentClass,
//            Param->Location,
//            FRotator::ZeroRotator
//        );
//        if (Env)
//        {
//            Env->InitEnv(Param);
//            Environments.Add(Env);
//        }
//        else
//        {
//            UE_LOG(LogTemp, Error, TEXT("AVectorEnvironment::InitEnv - Failed to spawn environment actor!"));
//        }
//    }
//
//    // Now parse the action space from the config 
//    ParseActionSpaceFromConfig();
//}
//
//TArray<FState> AVectorEnvironment::ResetEnv(int NumAgents)
//{
//    CurrentAgents = NumAgents;
//    TArray<FState> States;
//
//    for (ABaseEnvironment* Env : Environments)
//    {
//        FState S = Env->ResetEnv(CurrentAgents);
//        States.Add(S);
//    }
//    CurrentStates = States;
//    return CurrentStates;
//}
//
//TTuple<TArray<float>, TArray<float>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>>
//AVectorEnvironment::Transition()
//{
//    TArray<float> Dones;
//    TArray<float> Truncs;
//    TArray<float> Rewards;
//    TArray<FState> States;
//
//    LastStates = CurrentStates;
//
//    for (int32 i = 0; i < Environments.Num(); i++)
//    {
//        ABaseEnvironment* Env = Environments[i];
//        Env->PreTransition();
//
//        bool bDone = Env->Done();
//        bool bTrunc = Env->Trunc();
//        float RewardVal = Env->Reward();
//
//        Dones.Add((float)bDone);
//        Truncs.Add((float)bTrunc);
//        Rewards.Add(RewardVal);
//
//        if (bDone || bTrunc)
//        {
//            // if environment is done or truncated, we do a reset
//            States.Add(Env->ResetEnv(CurrentAgents));
//        }
//        else
//        {
//            States.Add(Env->State());
//        }
//
//        Env->PostTransition();
//    }
//
//    CurrentStates = States;
//    CurrentDones = Dones;
//    CurrentTruncs = Truncs;
//
//    return TTuple<TArray<float>, TArray<float>, TArray<float>, TArray<FAction>, TArray<FState>, TArray<FState>>(
//        Dones, Truncs, Rewards, LastActions, LastStates, States
//    );
//}
//
//void AVectorEnvironment::Step(TArray<FAction> Actions)
//{
//    if (Actions.Num() != Environments.Num())
//    {
//        UE_LOG(LogTemp, Warning, TEXT("AVectorEnvironment::Step - Mismatch in action count vs. environment count."));
//    }
//
//    for (int32 i = 0; i < Environments.Num(); i++)
//    {
//        // When action repeat is enabled for Runner, prevent from stepping done/truncated environemtns until call to reset
//        if (!CurrentDones[i] && !CurrentTruncs[i])
//        {
//            ABaseEnvironment* Env = Environments[i];
//            Env->PreStep();
//            Env->Act(Actions[i]);
//            Env->PostStep();
//        }
//        else {
//            int test = 0;
//        }
//    }
//
//    LastActions = Actions;
//}
//
//TArray<FState> AVectorEnvironment::GetStates()
//{
//    TArray<FState> TmpStates;
//    for (int32 i = 0; i < Environments.Num(); i++)
//    {
//        TmpStates.Add(Environments[i]->State());
//    }
//    CurrentStates = TmpStates;
//    return TmpStates;
//}
//
//TArray<FAction> AVectorEnvironment::SampleActions()
//{
//    TArray<FAction> Actions;
//
//    for (int32 envIndex = 0; envIndex < Environments.Num(); envIndex++)
//    {
//        FAction Sampled = EnvSample();
//        Actions.Add(Sampled);
//    }
//
//    return Actions;
//}
//
//void AVectorEnvironment::ParseActionSpaceFromConfig()
//{
//    DiscreteActionSizes.Empty();
//    ContinuousActionRanges.Empty();
//
//    if (!EnvConfig)
//    {
//        UE_LOG(LogTemp, Warning, TEXT("AVectorEnvironment::ParseActionSpaceFromConfig - EnvConfig is null."));
//        return;
//    }
//
//    // Example: We'll read from "environment/shape/action/agent/discrete"
//    // If it exists, parse each object with "num_choices"
//
//    if (EnvConfig->HasPath(TEXT("environment/shape/action/agent/discrete")))
//    {
//        // We need to parse an array of objects, e.g.  [ { "num_choices": 4 }, { "num_choices": 4 } ]
//        UEnvironmentConfig* DiscreteNode = EnvConfig->Get(TEXT("environment/shape/action/agent/discrete"));
//        TArray<UEnvironmentConfig*> DiscreteArray = DiscreteNode->AsArrayOfConfigs(); // This is a hypothetical method you must define
//
//        for (UEnvironmentConfig* Item : DiscreteArray)
//        {
//            // Each item is an object with "num_choices"
//            int32 NumChoices = Item->Get(TEXT("num_choices"))->AsInt();
//            DiscreteActionSizes.Add(NumChoices);
//        }
//    }
//
//    // Similarly, we look for "environment/shape/action/agent/continuous"
//    // e.g. [ [ -1.0, 1.0 ], [ 0, 10 ] ]
//    if (EnvConfig->HasPath(TEXT("environment/shape/action/agent/continuous")))
//    {
//        UEnvironmentConfig* ContinuousNode = EnvConfig->Get(TEXT("environment/shape/action/agent/continuous"));
//        TArray<UEnvironmentConfig*> ContinuousArray = ContinuousNode->AsArrayOfConfigs(); // also hypothetical
//
//        for (UEnvironmentConfig* RangeItem : ContinuousArray)
//        {
//            ContinuousActionRanges.Add(
//                {
//                    RangeItem->Get(TEXT("min"))->AsNumber(),
//                    RangeItem->Get(TEXT("max"))->AsNumber()
//                }
//            );
//        }
//    }
//}
//
//FAction AVectorEnvironment::EnvSample()
//{
//    FAction SampledAction;
//
//    // For each "agent" within this environment (multi-agent):
//    for (int32 agentIdx = 0; agentIdx < CurrentAgents; agentIdx++)
//    {
//        // 1) Discrete actions
//        for (int32 i = 0; i < DiscreteActionSizes.Num(); i++)
//        {
//            int32 RangeMax = DiscreteActionSizes[i];
//            int32 RandomChoice = FMath::RandRange(0, RangeMax - 1);
//            SampledAction.Values.Add(static_cast<float>(RandomChoice));
//        }
//
//        // 2) Continuous actions
//        for (int32 j = 0; j < ContinuousActionRanges.Num(); j++)
//        {
//            FVector2D Range = ContinuousActionRanges[j];
//            float RandomVal = FMath::RandRange(Range.X, Range.Y);
//            SampledAction.Values.Add(RandomVal);
//        }
//    }
//
//    return SampledAction;
//}
