using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json.Linq;
using System.IO;
using System.Text;
using System;

public class JsonReader : MonoBehaviour
{
    public List<string> frames = new List<string>();
    public List<float[]> RotJoints = new List<float[]>();
    string path = "C:/Users/sunxiaohan01/Desktop/sxh_git/human_dynamics/demo_output/davis-hike/rot_output/rot_output.json";
    public Dictionary<string, int> _boneNameToJointIndex = new Dictionary<string, int>();

    [ContextMenu("Read Json")]
    public void ReadJson()
    {
        string jsonStr = File.ReadAllText(path, Encoding.Default);
        frames.Add(jsonStr);

        Debug.Log(jsonStr);

        JObject jo = JObject.Parse(jsonStr);

        string rot = "rot_";
        for (int i = 0; i < 24; i++)
        {
            string jointName = rot + string.Format("{0:D2}", i);
            // Debug.Log(jointName);
            // the jointName is in order
            RotJoints.Add(Rot9float(jo, jointName));
        }
    }
    [ContextMenu("Clear Data")]
    public void ClearData()
    {
        frames.Clear();
        RotJoints.Clear();
    }
    public void init()
    {
        if (_boneNameToJointIndex.Count > 0)
        {
            Debug.LogWarning("skip init joint Dict");
            return;
        }
        _boneNameToJointIndex.Add("m_avg_Pelvis", 0);
        _boneNameToJointIndex.Add("m_avg_L_Hip", 1);
        _boneNameToJointIndex.Add("m_avg_R_Hip", 2);
        _boneNameToJointIndex.Add("m_avg_Spine1", 3);
        _boneNameToJointIndex.Add("m_avg_L_Knee", 4);
        _boneNameToJointIndex.Add("m_avg_R_Knee", 5);
        _boneNameToJointIndex.Add("m_avg_Spine2", 6);
        _boneNameToJointIndex.Add("m_avg_L_Ankle", 7);
        _boneNameToJointIndex.Add("m_avg_R_Ankle", 8);
        _boneNameToJointIndex.Add("m_avg_Spine3", 9);
        _boneNameToJointIndex.Add("m_avg_L_Foot", 10);
        _boneNameToJointIndex.Add("m_avg_R_Foot", 11);
        _boneNameToJointIndex.Add("m_avg_Neck", 12);
        _boneNameToJointIndex.Add("m_avg_L_Collar", 13);
        _boneNameToJointIndex.Add("m_avg_R_Collar", 14);
        _boneNameToJointIndex.Add("m_avg_Head", 15);
        _boneNameToJointIndex.Add("m_avg_L_Shoulder", 16);
        _boneNameToJointIndex.Add("m_avg_R_Shoulder", 17);
        _boneNameToJointIndex.Add("m_avg_L_Elbow", 18);
        _boneNameToJointIndex.Add("m_avg_R_Elbow", 19);
        _boneNameToJointIndex.Add("m_avg_L_Wrist", 20);
        _boneNameToJointIndex.Add("m_avg_R_Wrist", 21);
        _boneNameToJointIndex.Add("m_avg_L_Hand", 22);
        _boneNameToJointIndex.Add("m_avg_R_Hand", 23);
    }
    void Awake()
    {
        //init();
    }

    float Rotfloat(JObject jo, string jointName, int index)
    {
        float temp = -1.0f;
        if (float.TryParse(jo[jointName][index].ToString(), out temp))
        { }
        else
            Debug.LogError("can't convert to float or rot_index not exist!");
        return temp;
    }

    float[] Rot9float(JObject jo,string jointName)
    {
        float[] result = new float[9];
        for (int i = 0; i < 9; i++) 
        {
            result[i] = Rotfloat(jo, jointName, i);
        }

        return result;
    }
}
