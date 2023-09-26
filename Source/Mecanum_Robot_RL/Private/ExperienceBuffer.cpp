#include "ExperienceBuffer.h"

UExperienceBuffer::UExperienceBuffer() : BufferCapacity(0) {}

void UExperienceBuffer::AddExperiences(const TArray<FExperienceBatch>& EnvironmentTrajectories)
{
    if (ExperienceDeques.Num() == 0)
    {
        for (int32 i = 0; i < EnvironmentTrajectories[0].Experiences.Num(); ++i)
        {
            ExperienceDeques.Add(std::deque<FExperience>());
        }
    }

    for (int32 i = 0; i < EnvironmentTrajectories.Num(); ++i)
    {
        for (const FExperience& Exp : EnvironmentTrajectories[i].Experiences)
        {
            ExperienceDeques[i].push_back(Exp);
        }
    }

    EnsureBufferLimit();
}

TArray<FExperienceBatch> UExperienceBuffer::SampleExperiences(int32 BatchSize)
{
    TArray<FExperienceBatch> SampledExperiences;

    for (std::deque<FExperience>& Deque : ExperienceDeques)
    {
        FExperienceBatch Batch;

        if (Deque.empty() || (Deque.size() < (size_t)BatchSize))
        {
            return TArray<FExperienceBatch>();
        }

        for (int32 i = 0; i < BatchSize; ++i)
        {
            Batch.Experiences.Add(Deque.front());
            Deque.pop_front();
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

void UExperienceBuffer::EnsureBufferLimit()
{
    for (std::deque<FExperience>& Deque : ExperienceDeques)
    {
        while (!Deque.empty() && (Deque.size() > (size_t)BufferCapacity))
        {
            Deque.pop_front();
        }
    }
}
