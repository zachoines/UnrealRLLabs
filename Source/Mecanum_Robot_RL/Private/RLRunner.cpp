#include "RLRunner.h"


ARLRunner::ARLRunner()
{
    // Enable ticking
    PrimaryActorTick.bCanEverTick = true;
    ExperienceBufferInstance = NewObject<UExperienceBuffer>();
}

void ARLRunner::InitRunner(
    TSubclassOf<ABaseEnvironment> EnvironmentClass, 
    TArray<FBaseInitParams*> ParamsArray, 
    int BufferSize,
    int BatchSize,
    int NumEnvironments,
    int StateSize,
    int NumActions
)
{
    CurrentStep = 0;
    VectorEnvironment = GetWorld()->SpawnActor<AVectorEnvironment>(AVectorEnvironment::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);
    VectorEnvironment->InitEnv(EnvironmentClass, ParamsArray);

    AgentComm = NewObject<USharedMemoryAgentCommunicator>(this);
    Config.NumEnvironments = NumEnvironments;
    Config.NumActions = NumActions;
    Config.StateSize = StateSize;
    Config.BatchSize = BatchSize;
    AgentComm->Init(Config);

    VectorEnvironment->ResetEnv();
    ExperienceBufferInstance->SetBufferCapacity(BufferSize);
}

TArray<FAction> ARLRunner::GetActions(TArray<FState> States)
{
    if (AgentComm) {
        return AgentComm->GetActions(States);
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

    if (CurrentStep > 0) {
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

        if ((CurrentStep % Config.BatchSize) == 0) {
            TArray<FExperienceBatch> Transitions = ARLRunner::SampleExperiences(Config.BatchSize);
            AgentComm->Update(Transitions);
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
