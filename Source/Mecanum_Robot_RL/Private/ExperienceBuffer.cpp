#include "ExperienceBuffer.h"

UExperienceBuffer::UExperienceBuffer() : BufferCapacity(0) {}

void UExperienceBuffer::AddExperiences(const TArray<FExperienceBatch>& EnvironmentTrajectories)
{
    if (ExperienceDeques.Num() == 0)
    {
        for (int32 i = 0; i < EnvironmentTrajectories[0].Experiences.Num(); ++i)
        {
            ExperienceDeques.Add(TArray<FExperience>());
        }
    }

    // For each step in trajectory
    for (int32 i = 0; i < EnvironmentTrajectories.Num(); ++i)
    {
        // For each environment 
        for (int32 j = 0; j < EnvironmentTrajectories[i].Experiences.Num(); ++j)
        {
            // Append step in trajectory
            ExperienceDeques[j].Add(EnvironmentTrajectories[i].Experiences[j]);
        }
    }

    EnsureBufferLimit();
}

TArray<FExperienceBatch> UExperienceBuffer::SampleExperiences(int32 BatchSize)
{
    TArray<FExperienceBatch> SampledExperiences;

    for (TArray<FExperience>& Deque : ExperienceDeques)
    {
        FExperienceBatch Batch;

        if (Deque.Num() == 0 || Deque.Num() < BatchSize)
        {
            return TArray<FExperienceBatch>();
        }

        for (int32 i = 0; i < BatchSize; ++i)
        {
            Batch.Experiences.Add(Deque[0]);
            Deque.RemoveAt(0);
        }

        SampledExperiences.Add(Batch);
    }

    return SampledExperiences;
}

void UExperienceBuffer::SetBufferCapacity(int32 NewCapacity)
{
    BufferCapacity = NewCapacity;
    EnsureBufferLimit();
}

int32 UExperienceBuffer::Size()
{
    return ExperienceDeques[0].Num();
}

void UExperienceBuffer::EnsureBufferLimit()
{
    for (TArray<FExperience>& Deque : ExperienceDeques)
    {
        while (Deque.Num() > BufferCapacity)
        {
            Deque.RemoveAt(0);
        }
    }
}
