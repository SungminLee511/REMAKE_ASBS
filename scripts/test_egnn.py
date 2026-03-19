"""Quick EGNN equivariance & translation invariance test."""
import torch
import sys
sys.path.insert(0, "/home/RESEARCH/REMAKE_ASBS")
from asbs.models.egnn import EGNN

def random_rotation_matrix(device):
    """Random 3D rotation via QR decomposition."""
    A = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    # Ensure det(Q) = +1
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def test_equivariance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EGNN(n_particles=4, coord_dim=3, hidden_dim=64, n_layers=3).to(device)
    model.eval()

    B = 8
    t = torch.rand(B, device=device) * 0.998 + 0.001
    x = torch.randn(B, 12, device=device)  # 4 particles * 3 coords

    with torch.no_grad():
        out1 = model(t, x)  # [B, 12]

    # Apply rotation
    R = random_rotation_matrix(device)
    x_rot = x.view(B, 4, 3) @ R.T
    x_rot = x_rot.view(B, 12)

    with torch.no_grad():
        out2 = model(t, x_rot)

    # out2 should be R @ out1
    out1_rot = out1.view(B, 4, 3) @ R.T
    out1_rot = out1_rot.view(B, 12)

    err = (out2 - out1_rot).abs().max().item()
    print(f"Equivariance error: {err:.2e}")
    assert err < 1e-4, f"FAILED: equivariance error {err}"
    print("PASSED: Equivariance test")

    # Translation invariance
    shift = torch.randn(1, 1, 3, device=device) * 5
    x_shifted = (x.view(B, 4, 3) + shift).view(B, 12)

    with torch.no_grad():
        out3 = model(t, x_shifted)

    err_trans = (out3 - out1).abs().max().item()
    print(f"Translation invariance error: {err_trans:.2e}")
    assert err_trans < 1e-4, f"FAILED: translation error {err_trans}"
    print("PASSED: Translation invariance test")

if __name__ == "__main__":
    test_equivariance()
    print("\nAll EGNN tests passed!")
