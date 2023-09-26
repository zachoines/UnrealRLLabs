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

    VectorEnvironment = GetWorld()->SpawnActor<AVectorEnvironment>(AVectorEnvironment::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);
    VectorEnvironment->InitEnv(EnvironmentClass, ParamsArray);

    AgentComm = NewObject<USharedMemoryAgentCommunicator>(this);
    Config.NumEnvironments = NumEnvironments;
    Config.NumActions = NumActions;
    Config.StateSize = StateSize;
    Config.BatchSize = BatchSize;
    AgentComm->Init(Config);

    CurrentStates = VectorEnvironment->ResetEnv();
    ExperienceBufferInstance->SetBufferCapacity(BufferSize);
  
}

TArray<FAction> ARLRunner::GetActions(TArray<FState> States)
{
    // Get actions for the dummy states.
    if (AgentComm) {
        TArray<FAction> Actions = AgentComm->GetActions(States);

        // For now, just print the first action of the first environment to see if it works.
        if (Actions.Num() > 0 && Actions[0].Values.Num() > 0)
        {
            AgentComm->PrintActionsAsMatrix(Actions);
        }
        return Actions;
    }
    else {
        return VectorEnvironment->SampleActions();
    }
}

void ARLRunner::Tick(float DeltaTime)
{

    TArray<FAction> Actions = GetActions(CurrentStates);
    TTuple<TArray<bool>, TArray<float>, TArray<FState>> Result = VectorEnvironment->Step(Actions);

    TArray<FExperienceBatch> EnvironmentTrajectories; 
    
    // TODO: Right now we take only a single step at a time in each environment
    FExperienceBatch Batch;
    for (int32 i = 0; i < CurrentStates.Num(); i++)
    {
        FExperience Experience;
        Experience.State = CurrentStates[i];
        Experience.Action = Actions[i];
        Experience.Reward = Result.Get<1>()[i];
        Experience.NextState = Result.Get<2>()[i];
        Experience.Done = Result.Get<0>()[i];
        Batch.Experiences.Add(Experience);
    }

    EnvironmentTrajectories.Add(Batch);
    CurrentStates = Result.Get<2>();

    AddExperiences(EnvironmentTrajectories);
}

void ARLRunner::AddExperiences(const TArray<FExperienceBatch>& AllExperiences)
{
    ExperienceBufferInstance->AddExperiences(AllExperiences);
}

TArray<FExperienceBatch> ARLRunner::SampleExperiences(int bSize)
{
    return ExperienceBufferInstance->SampleExperiences(bSize);
}
