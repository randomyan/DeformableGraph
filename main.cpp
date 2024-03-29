#include <GL\freeglut.h>
//#include <GL\glext.h>
#include <math.h>
#define _USE_MATH_DEFINES

#include <flann/flann.hpp>
#include "TriMesh.h"
#include "DeformGraph.h"


using namespace std;
using Eigen::Vector3d;
/* ascii codes for special keys */
#define ESCAPE 27

/**********************************************************************
* Configuration
**********************************************************************/

#define INITIAL_WIDTH 600
#define INITIAL_HEIGHT 600
#define INITIAL_X_POS 100
#define INITIAL_Y_POS 100

#define WINDOW_NAME     "Deformation Gragh"
/**********************************************************************
* Globals
**********************************************************************/

GLsizei g_window_width;
GLsizei g_window_height;

const char* src_filename = "D:/project/DeformGraph/data/volume2.ply";
const char* dst_filename = "D:/project/DeformGraph/data/volume3.ply";

TriMesh mesh_in(src_filename);
TriMesh mesh_out(dst_filename);
TriMesh source_mesh(src_filename);
TriMesh d_mesh(src_filename);

DGraph graph;
//////////////////build graph para


/////////////////////////

Vector3d g_view_dir;
Vector3d g_view_pos;
float g_view_angle = M_PI_2;
float g_angle = M_PI;
Vector3d g_object_delta(0.0, 0.0, 0.0);
float g_scale = 1.0;
float g_rot_matrix[16];
bool g_mouse_down;
Vector3d g_mouse_pos(-1.0f, -1.0f, -1.0f);


/**********************************************************************
* Set the new size of the window
**********************************************************************/

void resize_scene(GLsizei width, GLsizei height)
{
	glViewport(0, 0, width, height);  /* reset the current viewport and
									  * perspective transformation */
	g_window_width = width;
	g_window_height = height;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	/* for the purposes of this assignment, we are making the world
	* coordinate system have a 1-1 correspondence with screen space
	* (in pixels).  The origin maps to the lower-left corner of the screen.*/
	gluPerspective(90.0, width / height, 0.1, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glGetFloatv(GL_MODELVIEW_MATRIX, g_rot_matrix);
}

/**********************************************************************
* any program initialization (set initial OpenGL state,
**********************************************************************/
void init()
{
	//init graph
	std::vector<int> sampling_index;
	double sampling_dist = 0.03;
//	double sampling_dist = 1;
	mesh_sampling(mesh_in, sampling_index, sampling_dist);
	graph.build_graph(mesh_in, sampling_index, 2.0*sampling_dist, 8);
	
	//deform graph
	double *raw_dataset = new double[3 * mesh_out.vertex_coord.size()];
	for (size_t i = 0; i<mesh_out.vertex_coord.size(); ++i) {
		raw_dataset[3 * i + 0] = mesh_out.vertex_coord[i][0];
		raw_dataset[3 * i + 1] = mesh_out.vertex_coord[i][1];
		raw_dataset[3 * i + 2] = mesh_out.vertex_coord[i][2];
	}
	
	flann::Matrix<double> flann_dataset(raw_dataset, mesh_out.vertex_coord.size(), 3);
	flann::Index< flann::L2<double> > flann_index(flann_dataset, flann::KDTreeIndexParams(1));
	flann_index.buildIndex();
	flann::Matrix<double>   query_node(new double[3], 1, 3);	// num of querys * dimension
	flann::Matrix<int>      indices_node(new int[1], 1, 1);		// num of querys * knn
	flann::Matrix<double>   dists_node(new double[1], 1, 1);

	double max_dist_neigh = 2.0*sampling_dist;
	double pre_energy = DBL_MAX;
	int iter = 0;
	do{
		// 1. build nearest correspondence from graph to mesh_out
		std::vector<int> corres_indexs; std::vector<Eigen::Vector3d> corres_constraints;
		for (size_t i = 0; i < graph.node_pos.size(); ++i) {
			query_node[0][0] = graph.node_pos[i][0]; query_node[0][1] = graph.node_pos[i][1]; query_node[0][2] = graph.node_pos[i][2];
			flann_index.knnSearch(query_node, indices_node, dists_node, 1, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
			if (dists_node[0][0] > max_dist_neigh*max_dist_neigh) continue;	//TODO: replace with a stronger condition for matching pairs
			corres_indexs.push_back(i);
			corres_constraints.push_back(mesh_out.vertex_coord[indices_node[0][0]]);
		}
		dprintf(stdout, "%d:\tfind pairs %d\n", iter, corres_indexs.size());
		//	cout << "iter:  " << iter << " corresp size  " << corres_indexs.size() << endl;

		// 2. use the correspondence to initialize a Deformer and call optimize_once and update graph
		double cur_energy = optimize_once(graph, corres_indexs, corres_constraints);
		//	dprintf(stdout, "min_energy = %lf\n", cur_energy);
		cout << "min_energy =  " << cur_energy << endl;
		if (fabs(pre_energy - cur_energy) < 1e-5) iter = 51;//break;
		pre_energy = cur_energy;
	} while (++iter < 1);
	graph.deform(source_mesh);


	// setting  view
	g_view_dir = Vector3d(0.0, 0.0, -1.0);
	mesh_in.getBoundingBox(mesh_in.BoundingBox_Min, mesh_in.BoundingBox_Max);
	Vector3d center = mesh_in.center;
	g_view_pos.x() = center.x();
	g_view_pos.y() = center.y();

	float model_height = mesh_in.BoundingBox_Max.y() - mesh_in.BoundingBox_Min.y();
	// want the object to take up 1/4 the view in y
	// so the total view angle is 4*model_height and 2*model_height is the height of the upper or lower triangle
	// tan(g_view_angle/2.0f) = model_height*2/zdist => zdist = model_height*2/tan(g_view_angle/2.0);
	g_view_pos.z() = mesh_in.BoundingBox_Max.z() + model_height * 2 / tanf(g_view_angle / 2.0f);
}


/**********************************************************************
* The main drawing functions.
**********************************************************************/
void draw_scene(void)
{
	/* clear the screen and the depth buffer */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	/* reset modelview matrix */
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	/*

	GLfloat lightPos0[] = { 520.0f, 400.0f, 504.0f, 0.0f };
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);

	GLfloat lightPos1[] = { 20.0f, 400.0f, -504.0f, 1.0f };
	glLightfv(GL_LIGHT1, GL_POSITION, lightPos1);
	*/
	gluLookAt(g_view_pos.x(), g_view_pos.y(), g_view_pos.z(),
		g_view_pos.x() + g_view_dir.x(), g_view_pos.y() + g_view_dir.y(), g_view_pos.z() + g_view_dir.z(),
		0.0f, 1.0f, 0.0f);

	glTranslatef(g_object_delta.x(), g_object_delta.y(), g_object_delta.z());
	glTranslatef(mesh_in.center[0], mesh_in.center[1], mesh_in.center[2]);
	glMultMatrixf(g_rot_matrix);
	glScalef(g_scale, g_scale, g_scale);
	glTranslatef(-mesh_in.center[0], -mesh_in.center[1], -mesh_in.center[2]);

//	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
//	glColor3f(1.0, 0.0, 0.0);
//	mesh_in.render(POINT_MODE);
	glColor3f(0.0,1.0,0.0);
	source_mesh.render(POINT_MODE);
//	glColor3f(0.0, 0.0, 1.0);
//	mesh_out.render(POINT_MODE);
//	glColor3f(1.0, 1.0, 0.0);
//	d_mesh.render(POINT_MODE);

	/* since this is double buffered, swap the
	* buffers to display what just got drawn */
	glutSwapBuffers();
}

void key_press(unsigned char key, int x, int y)
{
	Vector3d perp_dir;
	Vector3d c(0.0, 1.0, 0.0);
	float speed = 0.1;
	float rot_speed = 8.0;

	// Get u,v,n vectors where n is the direction we are looking
	// so it's perpendicular to the screen, and u and v are parallel
	// to the screen.  So u and v are what we want to use as the
	// rotation axes - u is the axis for mouse motion in the y direction
	// and v is the rotation axis for mouse motion in the x direction
	Vector3d n = -g_view_dir;
	Vector3d u = Vector3d(0.0f, 1.0f, 0.0f).cross(n);
	u.normalize();
	Vector3d v = n.cross(u);
	v.normalize();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	switch (key) {
	case 'w':
		g_view_pos += g_view_dir*speed;
		break;
	case 'a':
		g_view_pos += c.cross(g_view_dir)*speed;
		break;
	case 's':
		g_view_pos -= g_view_dir*speed;
		break;
	case 'd':
		g_view_pos += g_view_dir.cross(c)*speed;
		break;
	case 'i':  // rotate up
		glRotatef(-rot_speed, u.x(), u.y(), u.z());
		glMultMatrixf(g_rot_matrix);
		glGetFloatv(GL_MODELVIEW_MATRIX, g_rot_matrix);
		break;
	case 'j':  // rotate left
		glRotatef(-rot_speed, v.x(), v.y(), v.z());
		glMultMatrixf(g_rot_matrix);
		glGetFloatv(GL_MODELVIEW_MATRIX, g_rot_matrix);
		break;
	case 'k':  // rotate right
		glRotatef(rot_speed, v.x(), v.y(), v.z());
		glMultMatrixf(g_rot_matrix);
		glGetFloatv(GL_MODELVIEW_MATRIX, g_rot_matrix);
		break;
	case 'm':  // rotate down
		glRotatef(rot_speed, u.x(), u.y(), u.z());
		glMultMatrixf(g_rot_matrix);
		glGetFloatv(GL_MODELVIEW_MATRIX, g_rot_matrix);
		break;
	case 'y':
		g_object_delta += g_view_dir*speed;
		break;
	case 'g':
		g_object_delta += c.cross(g_view_dir)*speed;
		break;
	case 'h':
		g_object_delta += g_view_dir.cross(c)*speed;
		break;
	case 'b':
		g_object_delta -= g_view_dir*speed;
		break;
	case '+':
		g_scale += 0.1f;
		break;
	case '-':
		g_scale -= 0.1f;
		break;
	
	case '8':
	{//experiment
		//mesh_out = mesh_in;
		//init graph
		std::vector<int> sampling_index;
		double sampling_dist = 0.03;
		//double sampling_dist = 2;
		mesh_sampling(mesh_in, sampling_index, sampling_dist);
		graph.build_graph(mesh_in, sampling_index, 2.0*sampling_dist, 8);

		//deform graph
		double *raw_dataset = new double[3 * source_mesh.vertex_coord.size()];
		for (size_t i = 0; i<source_mesh.vertex_coord.size(); ++i) {
			raw_dataset[3 * i + 0] = source_mesh.vertex_coord[i][0];
			raw_dataset[3 * i + 1] = source_mesh.vertex_coord[i][1];
			raw_dataset[3 * i + 2] = source_mesh.vertex_coord[i][2];
		}

		flann::Matrix<double> flann_dataset(raw_dataset, source_mesh.vertex_coord.size(), 3);
		flann::Index< flann::L2<double> > flann_index(flann_dataset, flann::KDTreeIndexParams(1));
		flann_index.buildIndex();
		flann::Matrix<double>   query_node(new double[3], 1, 3);	// num of querys * dimension
		flann::Matrix<int>      indices_node(new int[1], 1, 1);		// num of querys * knn
		flann::Matrix<double>   dists_node(new double[1], 1, 1);

		double max_dist_neigh = 2.0*sampling_dist;
		double pre_energy = DBL_MAX;
		int iter = 0;
		do{
			// 1. build nearest correspondence from graph to mesh_out
			std::vector<int> corres_indexs; std::vector<Eigen::Vector3d> corres_constraints;
			for (size_t i = 0; i < graph.node_pos.size(); ++i) {
				query_node[0][0] = graph.node_pos[i][0]; query_node[0][1] = graph.node_pos[i][1]; query_node[0][2] = graph.node_pos[i][2];
				flann_index.knnSearch(query_node, indices_node, dists_node, 1, flann::SearchParams(flann::FLANN_CHECKS_UNLIMITED));
				if (dists_node[0][0] > max_dist_neigh*max_dist_neigh) continue;	//TODO: replace with a stronger condition for matching pairs
				corres_indexs.push_back(i);
				corres_constraints.push_back(source_mesh.vertex_coord[indices_node[0][0]]);
			}
			dprintf(stdout, "%d:\tfind pairs %d\n", iter, corres_indexs.size());
			//	cout << "iter:  " << iter << " corresp size  " << corres_indexs.size() << endl;

			// 2. use the correspondence to initialize a Deformer and call optimize_once and update graph
			double cur_energy = optimize_once(graph, corres_indexs, corres_constraints);
			//	dprintf(stdout, "min_energy = %lf\n", cur_energy);
			cout << "min_energy =  " << cur_energy << endl;
			if (fabs(pre_energy - cur_energy) < 1e-5) iter = 51;//break;
			pre_energy = cur_energy;
		} while (++iter < 1);
		graph.deform(d_mesh);
	}
	break;
	case '2':
	//	graph.deform(source_mesh);
		break;
	case ESCAPE: /* exit the program...normal termination. */
		exit(0);
	default:
		break;
	}

	glPopMatrix();

	glutPostRedisplay();

}

/**********************************************************************
* this function is called whenever the mouse is moved
**********************************************************************/


void handle_mouse_motion(int x, int y)
{
	Vector3d pos(x, y, 0.0);

	//
	if (g_mouse_down && g_mouse_pos.x() > 0.0f) {
		// Get u,v,n vectors where n is the direction we are looking
		// so it's perpendicular to the screen, and u and v are parallel
		// to the screen.  So u and v are what we want to use as the
		// rotation axes - u is the axis for mouse motion in the y direction
		// and v is the rotation axis for mouse motion in the x direction
		Vector3d n = -g_view_dir;
		Vector3d u = Vector3d(0.0f, 1.0f, 0.0f).cross(n);
		u.normalize();
		Vector3d v = n.cross(u);
		v.normalize();

		// rotation axis is a linear combination of up and right based on mouse motion
		Vector3d delta = pos - g_mouse_pos;
		float delta_len = delta.norm();

		if (delta_len > 0.0001f) {
			Vector3d rot_axis = v*delta.x() + u*delta.y();
			rot_axis.normalize();

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();

			glRotatef(delta_len*0.5f, rot_axis.x(), rot_axis.y(), rot_axis.z());
			glMultMatrixf(g_rot_matrix);
			glGetFloatv(GL_MODELVIEW_MATRIX, g_rot_matrix);

			//cout << delta_len << " " << rot_axis.x() << " " << rot_axis.y() << " " << rot_axis.z() << endl;

			glPopMatrix();
			glutPostRedisplay();
		}
	}

	g_mouse_pos = pos;
}

/**********************************************************************
* this function is called whenever a mouse button is pressed or released
**********************************************************************/

void handle_mouse_click(int btn, int state, int x, int y)
{
	// Otherwise when window gets focus, object may jump (big delta)
	g_mouse_pos = Vector3d(x, y, 0.0);

	switch (btn) {
	case GLUT_LEFT_BUTTON:
		if (state == GLUT_DOWN)
			g_mouse_down = true;
		else
			g_mouse_down = false;
		break;
	}
}

void special_key(int key, int x, int y)
{
	float angle_delta = 0.02f;

	switch (key) {
	case GLUT_KEY_RIGHT: //right arrow
		g_angle -= 0.1f;
		break;
	case GLUT_KEY_LEFT: //left arrow
		g_angle += 0.1f;
		break;
	case GLUT_KEY_UP: //up arrow
		break;
	case GLUT_KEY_DOWN: //down arrow
		break;
	default:
		break;
	}


	g_view_dir = Vector3d(sin(g_angle), 0.0, cos(g_angle));

	glutPostRedisplay();

}


int main(int argc, char * argv[])
{

	/* Initialize GLUT */
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(INITIAL_WIDTH, INITIAL_HEIGHT);
	glutInitWindowPosition(INITIAL_X_POS, INITIAL_Y_POS);
	glutCreateWindow(WINDOW_NAME);

	/* Register callback functions */
	glutDisplayFunc(draw_scene);
	glutReshapeFunc(resize_scene);       //Initialize the viewport when the window size changes.
	glutKeyboardFunc(key_press);         //handle when the key is pressed
	glutMouseFunc(handle_mouse_click);   //check the Mouse Button(Left, Right and Center) status(Up or Down)
	glutMotionFunc(handle_mouse_motion); //Check the Current mouse position when mouse moves
	glutSpecialFunc(special_key);        //Special Keyboard Key fuction(For Arrow button and F1 to F10 button)

	/* Initialize GL */
	init();

	/* Enter event processing loop */
	glutMainLoop();

	return 1;
}

