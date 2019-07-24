using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class JointManager : MonoBehaviour
{
    public int frame = 0;
    public RotateTest[] rots;
    public GameObject _jsonReader;
    public GameObject _jointVisPrefab;
    // Start is called before the first frame update
    void Start()
    {
        //UpdateFrame();
    }
    [ContextMenu("Visualize Joint Toggle")]
    public void VisualizeJoint()
    {
        if (_jointVisPrefab == null)
            Debug.LogError("missing jointVisPrefab, please assign it");
        rots = GetComponentsInChildren<RotateTest>();
        if(rots == null)
            Debug.LogError("not RotateTest or Joint in Children!");
        foreach(var rot in rots)
        {
            if (rot.transform.Find("Joint_vis(Clone)") == null)
            {
                GameObject go = Instantiate(_jointVisPrefab, Vector3.zero, Quaternion.identity) as GameObject;
                go.transform.parent = rot.transform;
                go.transform.localPosition = Vector3.zero;
                go.transform.localRotation = rot.transform.localRotation;
            }
            else
            {
                GameObject go = rot.transform.Find("Joint_vis(Clone)").gameObject;
                DestroyImmediate(go);
            }
        }
    }



    [ContextMenu("Restore T pose")]
    public void RestoreTpose()
    {
        if (_jsonReader == null)
        {
            Debug.LogError("missing jsonReader, please assign it");
        }
        _jsonReader.GetComponent<JsonReader>().init();
        rots = GetComponentsInChildren<RotateTest>();
        if (rots == null)
            Debug.LogError("not RotateTest or Joint in Children!");
        foreach (var rot in rots)
        {
            int index = -1;
            try
            {
                index = _jsonReader.GetComponent<JsonReader>()._boneNameToJointIndex[rot.gameObject.name];
                //Debug.Log(rot.gameObject.name + "'s index is : " + index);
                if (index == -1)
                    Debug.LogError("index is -1, something gets wrong");
            }
            catch (Exception e)
            {
                Debug.LogError(e.Message);
            }          
            rot.localMat = Matrix4x4.identity;
            rot.UpdateTransform();
        }
    }


    [ContextMenu("Update Frame")]
    public void UpdateFrame()
    {
        if (frame != 0)
            return;

        if (_jsonReader == null)
        {
            Debug.LogError("missing jsonReader, please assign it");
            return;
        }
        _jsonReader.GetComponent<JsonReader>().init();
        rots = GetComponentsInChildren<RotateTest>();
        if (rots == null)
            Debug.LogError("not RotateTest or Joint in Children!");
        if (_jsonReader.GetComponent<JsonReader>().RotJoints.Count == 0)
        {
            Debug.LogError("RotJoints's count is zero, you probably havn't read json file");
            return;
        }
        foreach (var rot in rots)
        {
            int index = -1;
            try
            {
                index = _jsonReader.GetComponent<JsonReader>()._boneNameToJointIndex[rot.gameObject.name];
                Debug.Log(rot.gameObject.name + "'s index is : " + index);
                if (index == -1)
                    Debug.LogError("index is -1, something gets wrong");
            }
            catch (Exception e)
            {
                Debug.LogError(e.Message);
            }
            float[] rot9 = _jsonReader.GetComponent<JsonReader>().RotJoints[index];
            rot.AssignMat(rot9);
            rot.UpdateTransform();
        }
    }
}
