#include "RLRunner.h"
#include "EnvironmentConfig.h"
#include "Misc/Paths.h"

// Sets default values
ARLRunner::ARLRunner()
{
    PrimaryActorTick.bCanEverTick = true; // We tick each frame
    CurrentAgents = -1;

    // Create the experience buffer
    ExperienceBufferInstance = NewObject<UExperienceBuffer>();

    CurrentStep = 0;
    CurrentUpdate = 0;
    ActionRepeatCounter = 0;
}

void ARLRunner::InitRunner(
    TSubclassOf<ABaseEnvironment> EnvironmentClass,
    TArray<FBaseInitParams*> ParamsArray,
    UEnvironmentConfig* InEnvConfig
)
{
    EnvConfig = InEnvConfig;

    // 1) Check if environment/shape/agent path exists => multi-agent or not
    IsMultiAgent = EnvConfig->HasPath(TEXT("environment/shape/state/agent"));

    if (IsMultiAgent)
    {
        // If agent block exists, read min & max from there
        MinAgents = EnvConfig->Get(TEXT("environment/shape/state/agent/min"))->AsInt();
        MaxAgents = EnvConfig->Get(TEXT("environment/shape/state/agent/max"))->AsInt();
    }
    else
    {
        // If there's no agent block, single-agent
        MinAgents = 1;
        MaxAgents = 1;
    }

    // 2) Read training hyperparams from "train" section
    //    e.g. "train/buffer_size", "train/batch_size", "train/ActionRepeat"
    if (EnvConfig->HasPath(TEXT("train/buffer_size")))
    {
        BufferSize = EnvConfig->Get(TEXT("train/buffer_size"))->AsInt();
    }
    if (EnvConfig->HasPath(TEXT("train/batch_size")))
    {
        BatchSize = EnvConfig->Get(TEXT("train/batch_size"))->AsInt();
    }
    if (EnvConfig->HasPath(TEXT("train/ActionRepeat")))
    {
        ActionRepeat = EnvConfig->Get(TEXT("train/ActionRepeat"))->AsInt();
    }

    // 3) Spawn & init the vector environment
    VectorEnvironment = GetWorld()->SpawnActor<AVectorEnvironment>(
        AVectorEnvironment::StaticClass(),
        FVector::ZeroVector,
        FRotator::ZeroRotator
    );
    if (!VectorEnvironment)
    {
        UE_LOG(LogTemp, Error, TEXT("RLRunner::InitRunner - Failed to spawn AVectorEnvironment."));
        return;
    }
    VectorEnvironment->InitEnv(EnvironmentClass, ParamsArray);

    // 4) Create shared memory communicator & init with config
    AgentComm = NewObject<USharedMemoryAgentCommunicator>(this);
    if (AgentComm)
    {
        // Pass the entire config so the communicator can read e.g. state size, etc.
        AgentComm->Init(EnvConfig);
    }
    else
    {
        UE_LOG(LogTemp, Error, TEXT("RLRunner::InitRunner - Could not create SharedMemoryAgentCommunicator."));
    }

    // 5) Decide how many agents to use now
    CurrentAgents = (IsMultiAgent)
        ? FMath::RandRange(MinAgents, MaxAgents)
        : 1;

    // 6) Reset the environment w/ that agent count
    VectorEnvironment->ResetEnv(CurrentAgents);

    // 7) Setup experience buffer
    ExperienceBufferInstance->SetBufferCapacity(BufferSize);

    // Reset counters
    CurrentStep = 0;
    CurrentUpdate = 0;
    ActionRepeatCounter = 0;
}

void ARLRunner::Tick(float DeltaTime)
{
    // 1) Only record transitions once per "action" if ActionRepeat > 0
    if (ActionRepeatCounter == 0)
    {
        // VectorEnvironment->Transition() moves env forward 1 step 
        auto [Dones, Truncs, Rewards, LastActions, States, NextStates] = VectorEnvironment->Transition();

        // If not the first step, record experiences
        if (CurrentStep > 1)
        {
            TArray<FExperienceBatch> EnvironmentTrajectories;
            FExperienceBatch Batch;

            for (int32 i = 0; i < States.Num(); i++)
            {
                FExperience Exp;
                Exp.State = States[i];
                Exp.Action = LastActions[i];
                Exp.Done = (Dones[i] != 0.f);
                Exp.Trunc = (Truncs[i] != 0.f);
                Exp.Reward = Rewards[i];
                Exp.NextState = NextStates[i];
                Batch.Experiences.Add(Exp);
            }
            EnvironmentTrajectories.Add(Batch);
            AddExperiences(EnvironmentTrajectories);

            // If the replay buffer is sufficiently full, do training update
            if (ExperienceBufferInstance->Size() >= BatchSize)
            {
                CurrentUpdate++;
                TArray<FExperienceBatch> Transitions = SampleExperiences(BatchSize);
                /*AgentComm->WriteTransitionsToFile(
                    Transitions, 
                    "C:\\Users\\zachoines\\Documents\\Unreal\\UnrealRLLabs\\Content\\Python\\TEST\\UnrealTransitions.csv"
                );*/
                // AgentComm->Update(Transitions, CurrentAgents);
            }
        }

        // 2) Query new actions from the Python side
        Actions = GetActions(VectorEnvironment->GetStates(), Dones, Truncs);
    }

    // 3) Apply the actions to the environment
    VectorEnvironment->Step(Actions);
    CurrentStep++;

    // 4) Handle action-repeat logic
    if (ActionRepeat > 0)
    {
        ActionRepeatCounter = (ActionRepeatCounter + 1) % ActionRepeat;
    }
    else
    {
        ActionRepeatCounter = 0;
    }
}

TArray<FAction> ARLRunner::GetActions(TArray<FState> States, TArray<float> Dones, TArray<float> Truncs)
{
    if (true) // !AgentComm)
    {
        // fallback: sample random actions
        UE_LOG(LogTemp, Warning, TEXT("RLRunner::GetActions - AgentComm is null. Using random actions."));
        return VectorEnvironment->SampleActions();
    }

    return AgentComm->GetActions(States, Dones, Truncs, CurrentAgents);
}

void ARLRunner::AddExperiences(const TArray<FExperienceBatch>& EnvironmentTrajectories)
{
    if (!ExperienceBufferInstance)
    {
        UE_LOG(LogTemp, Warning, TEXT("RLRunner::AddExperiences - ExperienceBufferInstance is null."));
        return;
    }
    ExperienceBufferInstance->AddExperiences(EnvironmentTrajectories);
}

TArray<FExperienceBatch> ARLRunner::SampleExperiences(int bSize)
{
    if (!ExperienceBufferInstance)
    {
        UE_LOG(LogTemp, Warning, TEXT("RLRunner::SampleExperiences - ExperienceBufferInstance is null."));
        return {};
    }
    return ExperienceBufferInstance->SampleExperiences(bSize);
}