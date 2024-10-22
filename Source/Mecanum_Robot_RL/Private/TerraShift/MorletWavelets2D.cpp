#include "TerraShift/MorletWavelets2D.h"
#include "Math/UnrealMathUtility.h"

// Constructor
MorletWavelets2D::MorletWavelets2D(int32 InGridSizeX, int32 InGridSizeY, float InPhaseVelocity)
    : GridSizeX(InGridSizeX), GridSizeY(InGridSizeY), PhaseVelocity(InPhaseVelocity)
{
    Initialize();
}

// Initialize grid coordinates and heights
void MorletWavelets2D::Initialize()
{
    XGrid = Matrix2D(GridSizeY, GridSizeX);
    YGrid = Matrix2D(GridSizeY, GridSizeX);

    float XStep = 1.0f;
    float YStep = 1.0f;

    for (int32 i = 0; i < GridSizeY; ++i)
    {
        for (int32 j = 0; j < GridSizeX; ++j)
        {
            XGrid[i][j] = j * XStep;
            YGrid[i][j] = i * YStep;
        }
    }

    Heights = Matrix2D(GridSizeY, GridSizeX, 0.0f);
}

// Update function with height movement clamping
Matrix2D MorletWavelets2D::Update(const TArray<AgentParameters>& UpdatedParameters)
{
    Matrix2D NewHeights(GridSizeY, GridSizeX, 0.0f);

    for (const AgentParameters& Agent : UpdatedParameters)
    {
        float x_a = Agent.Position.X;
        float y_a = Agent.Position.Y;
        float A_a = Agent.Amplitude;
        float theta_a = Agent.WaveOrientation;
        float k_a = Agent.Wavenumber;
        float omega_a = Agent.Frequency;
        float phi_a = Agent.Phase;
        float sigma_a = Agent.Sigma;
        float Time = Agent.Time;

        Matrix2D XShifted = XGrid - x_a;
        Matrix2D YShifted = YGrid - y_a;

        float CosTheta = FMath::Cos(theta_a);
        float SinTheta = FMath::Sin(theta_a);

        Matrix2D XRot = (XShifted * CosTheta) + (YShifted * SinTheta);
        Matrix2D YRot = (XShifted * (-SinTheta)) + (YShifted * CosTheta);

        Matrix2D Envelope = ((XRot * XRot + YRot * YRot) / (-2.0f * sigma_a * sigma_a)).Exp();

        Matrix2D Phase = XRot * k_a - omega_a * Time + phi_a;
        Matrix2D Wave = Envelope * (Phase.Cos()) * A_a;

        NewHeights += Wave;
    }

    // Smooth transition by clamping height movement
    for (int32 i = 0; i < GridSizeY; ++i)
    {
        for (int32 j = 0; j < GridSizeX; ++j)
        {
            float DeltaHeight = FMath::Clamp(NewHeights[i][j] - Heights[i][j], -MaxDeltaHeight, MaxDeltaHeight);
            float MaxHeight = FMath::Clamp(NewHeights[i][j] - Heights[i][j], -MaxDeltaHeight, MaxDeltaHeight);
            Heights[i][j] += DeltaHeight;
        }
    }

    return Heights;
}

// Reset function
void MorletWavelets2D::Reset()
{
    Heights.Init(0.0f);
}

// Get the current height map
const Matrix2D& MorletWavelets2D::GetHeights() const
{
    return Heights;
}

// Get phase velocity
float MorletWavelets2D::GetPhaseVelocity() const
{
    return PhaseVelocity;
}
