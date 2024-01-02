#include "RLRunner.h"

ARLRunner::ARLRunner()
{
    // Enable ticking
    CurrentAgents = -1;
    PrimaryActorTick.bCanEverTick = true;
    ExperienceBufferInstance = NewObject<UExperienceBuffer>();
}

void ARLRunner::InitRunner(
    TSubclassOf<ABaseEnvironment> EnvironmentClass,
    TArray<FBaseInitParams*> ParamsArray,
    FTrainParams TrainParams
)
{
    TrainerParams = TrainParams;
    VectorEnvironment = GetWorld()->SpawnActor<AVectorEnvironment>(AVectorEnvironment::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);
    VectorEnvironment->InitEnv(EnvironmentClass, ParamsArray);

    CurrentStep = 0;
    CurrentUpdate = 0;
    TrainerParams.NumEnvironments = ParamsArray.Num();

    AgentComm = NewObject<USharedMemoryAgentCommunicator>(this);
    AgentComm->Init(VectorEnvironment->SingleEnvInfo, TrainParams);

    CurrentAgents = VectorEnvironment->SingleEnvInfo.IsMultiAgent ? FMath::RandRange(TrainParams.MinAgents, TrainParams.MaxAgents) : -1;
    VectorEnvironment->ResetEnv(CurrentAgents);
    ExperienceBufferInstance->SetBufferCapacity(TrainParams.BufferSize);
}

TArray<FAction> ARLRunner::GetActions(TArray<FState> States)
{
    if (AgentComm) {
        return AgentComm->GetActions(States, CurrentAgents);
    }
    else {
        return VectorEnvironment->SampleActions();
    }
}

void ARLRunner::Tick(float DeltaTime)
{
    auto [Dones, Truncs, Rewards, LastActions, LastStates, CurrentStates] = VectorEnvironment->Transition();
    TArray<FAction> Actions = GetActions(CurrentStates);
    VectorEnvironment->Step(Actions);

    if (CurrentStep > 1) {
        TArray<FExperienceBatch> EnvironmentTrajectories;
        FExperienceBatch Batch;
        for (int32 i = 0; i < CurrentStates.Num(); i++)
        {
            FExperience Experience;
            Experience.State = LastStates[i];
            Experience.Action = LastActions[i];
            Experience.Done = Dones[i];
            Experience.Trunc = Truncs[i];
            Experience.Reward = Rewards[i];
            Experience.NextState = CurrentStates[i];
            Batch.Experiences.Add(Experience);
        }

        EnvironmentTrajectories.Add(Batch);
        AddExperiences(EnvironmentTrajectories);

        if (ExperienceBufferInstance->Size() == TrainerParams.BatchSize) {
            CurrentUpdate += 1;
            TArray<FExperienceBatch> Transitions = ARLRunner::SampleExperiences(TrainerParams.BatchSize);
            AgentComm->Update(Transitions, CurrentAgents);

            if (VectorEnvironment->SingleEnvInfo.IsMultiAgent && (CurrentUpdate % TrainerParams.AgentsResetFrequency == 0)) {
                CurrentAgents = FMath::RandRange(TrainerParams.MinAgents, TrainerParams.MaxAgents);
                VectorEnvironment->ResetEnv(CurrentAgents);
                CurrentStep = 0;
            }
        }
    }

    CurrentStep += 1;
}

void ARLRunner::AddExperiences(const TArray<FExperienceBatch>& AllExperiences)
{
    ExperienceBufferInstance->AddExperiences(AllExperiences);
}

TArray<FExperienceBatch> ARLRunner::SampleExperiences(int bSize)
{
    return ExperienceBufferInstance->SampleExperiences(bSize);
}
