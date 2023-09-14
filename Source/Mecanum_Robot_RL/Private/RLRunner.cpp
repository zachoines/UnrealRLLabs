#include "RLRunner.h"
#include "BaseEnvironment.h"


ARLRunner::ARLRunner()
{
    // Enable ticking
    PrimaryActorTick.bCanEverTick = true;
    ExperienceBufferInstance = NewObject<UExperienceBuffer>();
}

void ARLRunner::InitRunner(TSubclassOf<ABaseEnvironment> EnvironmentClass, TArray<FBaseInitParams*> ParamsArray, int32 BufferSize, int32 BatchSize)
{
    this->buffSize = BufferSize;
    this->batchSize = BatchSize;

    VectorEnvironment = GetWorld()->SpawnActor<AVectorEnvironment>(AVectorEnvironment::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator);
    VectorEnvironment->InitEnv(EnvironmentClass, ParamsArray);

    States = VectorEnvironment->ResetEnv();
    ExperienceBufferInstance->SetBufferCapacity(buffSize);
}

TArray<TArray<float>> ARLRunner::GetActions(TArray<TArray<float>> State)
{
    // Placeholder implementation
    return VectorEnvironment->SampleActions();
}

void ARLRunner::Tick(float DeltaTime)
{

    TArray<TArray<float>> Actions = GetActions(States);
    TTuple<TArray<bool>, TArray<float>, TArray<TArray<float>>> Result = VectorEnvironment->Step(Actions);

    TArray<FExperienceBatch> EnvironmentTrajectories; 
    
    // TODO: Right now we take only a single step at a time in each environment
    FExperienceBatch Batch;
    for (int32 i = 0; i < States.Num(); i++)
    {
        FExperience Experience;
        Experience.State = States[i];
        Experience.Action = Actions[i];
        Experience.Reward = Result.Get<1>()[i];
        Experience.NextState = Result.Get<2>()[i];
        Experience.Done = Result.Get<0>()[i];
        Batch.Experiences.Add(Experience);
    }

    EnvironmentTrajectories.Add(Batch);

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
