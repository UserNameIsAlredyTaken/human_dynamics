using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RotateTest : MonoBehaviour
{
    public Matrix4x4 localMat = new Matrix4x4();
    // Start is called before the first frame update
    void Awake()
    {
        if (gameObject.name[0] == 'r' && gameObject.name[1]=='o')
        {
            return;
        }
        localMat = Matrix4x4.identity;
    }

    [ContextMenu("UpdateTransform")]
    public void UpdateTransform()
    {
        Quaternion origin = Quaternion.LookRotation(localMat.GetColumn(2), localMat.GetColumn(1));
        // Debug.Log(gameObject.name + "'s original Quatenion is: " + origin.ToString("F4"));

        // because SMPL is RHC but Unity is LHC so need to modify the Quaternion 
        Quaternion modified = new Quaternion(-origin.x, origin.y, origin.z, -origin.w);

        // Debug.Log(gameObject.name + "'s modified Quatenion is: " + modified.ToString("F4"));

        transform.localRotation = modified;
    }

    public void AssignMat(float[] rot9)
    {
        localMat[0, 0] = rot9[0];
        localMat[0, 1] = rot9[1];
        localMat[0, 2] = rot9[2];
        localMat[1, 0] = rot9[3];
        localMat[1, 1] = rot9[4];
        localMat[1, 2] = rot9[5];
        localMat[2, 0] = rot9[6];
        localMat[2, 1] = rot9[7];
        localMat[2, 2] = rot9[8];
    }


}
